/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.common.gbt;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.classification.gbtclassifier.GBTClassifier;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.gbt.defs.BoostingStrategy;
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.LossType;
import org.apache.flink.ml.common.gbt.defs.Node;
import org.apache.flink.ml.common.gbt.defs.TaskType;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.SparseVectorTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.regression.gbtregressor.GBTRegressor;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.functions.sink.DiscardingSink;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.analysis.function.Logit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/** Runs a gradient boosting trees implementation. */
public class GBTRunner {

    private static final Logger LOG = LoggerFactory.getLogger(GBTRunner.class);

    private static boolean isVectorType(TypeInformation<?> typeInfo) {
        return typeInfo instanceof DenseVectorTypeInfo
                || typeInfo instanceof SparseVectorTypeInfo
                || typeInfo instanceof VectorTypeInfo;
    }

    public static DataStream<GBTModelData> train(Table data, BaseGBTParams<?> estimator) {
        String[] featuresCols = estimator.getFeaturesCols();
        TypeInformation<?>[] featuresTypes =
                Arrays.stream(featuresCols)
                        .map(d -> TableUtils.getTypeInfoByName(data.getResolvedSchema(), d))
                        .toArray(TypeInformation[]::new);
        for (int i = 0; i < featuresCols.length; i += 1) {
            Preconditions.checkArgument(
                    null != featuresTypes[i],
                    String.format(
                            "Column name %s not existed in the input data.", featuresCols[i]));
        }

        boolean isInputVector = featuresCols.length == 1 && isVectorType(featuresTypes[0]);
        return train(data, getStrategy(estimator, isInputVector));
    }

    /** Trains a gradient boosting tree model with given data and parameters. */
    static DataStream<GBTModelData> train(Table dataTable, BoostingStrategy strategy) {
        Tuple2<Table, DataStream<FeatureMeta>> preprocessResult =
                strategy.isInputVector
                        ? Preprocess.preprocessVecCol(dataTable, strategy)
                        : Preprocess.preprocessCols(dataTable, strategy);
        dataTable = preprocessResult.f0;
        DataStream<FeatureMeta> featureMeta = preprocessResult.f1;
        DataStream<Double> prior = calcPrior(dataTable, strategy);
        return boost(dataTable, strategy, featureMeta, prior);
    }

    public static DataStream<Map<String, Double>> getFeatureImportance(
            DataStream<GBTModelData> modelData) {
        return modelData
                .map(
                        value -> {
                            Map<Integer, Double> featureImportanceMap = new HashMap<>();
                            for (int i = 0; i < value.featureNames.size(); i += 1) {
                                featureImportanceMap.put(i, 0.);
                            }
                            double sum = 0.;
                            for (List<Node> tree : value.allTrees) {
                                for (Node node : tree) {
                                    if (node.isLeaf) {
                                        continue;
                                    }
                                    featureImportanceMap.merge(
                                            node.split.featureId, node.split.gain, Double::sum);
                                    sum += node.split.gain;
                                }
                            }
                            if (sum > 0.) {
                                for (Map.Entry<Integer, Double> entry :
                                        featureImportanceMap.entrySet()) {
                                    entry.setValue(entry.getValue() / sum);
                                }
                            }

                            List<String> featureNames = value.featureNames;
                            return featureImportanceMap.entrySet().stream()
                                    .collect(
                                            Collectors.toMap(
                                                    d -> featureNames.get(d.getKey()),
                                                    Map.Entry::getValue));
                        })
                .returns(Types.MAP(Types.STRING, Types.DOUBLE));
    }

    private static DataStream<GBTModelData> boost(
            Table dataTable,
            BoostingStrategy strategy,
            DataStream<FeatureMeta> featureMeta,
            DataStream<Double> prior) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) dataTable).getTableEnvironment();

        final String featureMetaBcName = "featureMeta";
        final String priorBcName = "prior";
        Map<String, DataStream<?>> bcMap = new HashMap<>();
        bcMap.put(featureMetaBcName, featureMeta);
        bcMap.put(priorBcName, prior);

        DataStream<TrainContext> initTrainContext =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(
                                tEnv.toDataStream(tEnv.fromValues(0), Integer.class)),
                        bcMap,
                        (inputs) -> {
                            //noinspection unchecked
                            DataStream<Integer> input = (DataStream<Integer>) (inputs.get(0));
                            return input.map(
                                    new InitTrainContextFunction(
                                            featureMetaBcName, priorBcName, strategy));
                        });

        DataStream<Row> data = tEnv.toDataStream(dataTable);
        DataStreamList dataStreamList =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(
                                initTrainContext.broadcast(),
                                // adds a dummy variable stream in order to match the feedback
                                // stream
                                initTrainContext.flatMap(
                                        (v, out) -> {}, initTrainContext.getType())),
                        ReplayableDataStreamList.notReplay(data, featureMeta),
                        IterationConfig.newBuilder()
                                .setOperatorLifeCycle(IterationConfig.OperatorLifeCycle.ALL_ROUND)
                                .build(),
                        new BoostIterationBody(strategy));
        DataStream<Row> evalResult = dataStreamList.get(1);
        evalResult
                .map(
                        (value) -> {
                            System.out.println("!!!!" + value);
                            return value;
                        })
                .addSink(new DiscardingSink<>());
        return dataStreamList.get(0);
    }

    public static BoostingStrategy getStrategy(BaseGBTParams<?> estimator, boolean isInputVector) {
        final Map<Param<?>, Object> paramMap = estimator.getParamMap();
        final Set<Param<?>> unsupported =
                new HashSet<>(
                        Arrays.asList(
                                BaseGBTParams.WEIGHT_COL,
                                BaseGBTParams.LEAF_COL,
                                BaseGBTParams.VALIDATION_INDICATOR_COL));
        List<Param<?>> unsupportedButSet =
                unsupported.stream()
                        .filter(d -> null != paramMap.get(d))
                        .collect(Collectors.toList());
        if (!unsupportedButSet.isEmpty()) {
            throw new UnsupportedOperationException(
                    String.format(
                            "Parameters %s are not supported yet right now.",
                            unsupportedButSet.stream()
                                    .map(d -> d.name)
                                    .collect(Collectors.joining(", "))));
        }

        BoostingStrategy strategy = new BoostingStrategy();
        strategy.featuresCols = estimator.getFeaturesCols();
        strategy.isInputVector = isInputVector;
        strategy.labelCol = estimator.getLabelCol();
        strategy.categoricalCols = estimator.getCategoricalCols();

        strategy.maxDepth = estimator.getMaxDepth();
        strategy.maxBins = estimator.getMaxBins();
        strategy.maxCategoriesNum = estimator.getMaxCategoriesNum();
        strategy.minInstancesPerNode = estimator.getMinInstancesPerNode();
        strategy.minWeightFractionPerNode = estimator.getMinWeightFractionPerNode();
        strategy.minInfoGain = estimator.getMinInfoGain();
        strategy.maxIter = estimator.getMaxIter();
        strategy.stepSize = estimator.getStepSize();
        strategy.seed = estimator.getSeed();
        strategy.subsamplingRate = estimator.getSubsamplingRate();
        strategy.featureSubsetStrategy = estimator.getFeatureSubsetStrategy();
        strategy.regGamma = estimator.getRegGamma();
        strategy.regLambda = estimator.getRegLambda();
        strategy.baseScore = estimator.getBaseScore();
        strategy.numThreads = estimator.getNumThreads();

        String lossTypeStr;
        if (estimator instanceof GBTClassifier) {
            strategy.taskType = TaskType.CLASSIFICATION;
            lossTypeStr = ((GBTClassifier) estimator).getLossType();
        } else if (estimator instanceof GBTRegressor) {
            strategy.taskType = TaskType.REGRESSION;
            lossTypeStr = ((GBTRegressor) estimator).getLossType();
        } else {
            throw new IllegalArgumentException(
                    String.format(
                            "Unexpected type of estimator: %s.",
                            estimator.getClass().getSimpleName()));
        }
        strategy.lossType = LossType.valueOf(lossTypeStr.toUpperCase());
        strategy.maxNumLeaves = 1 << strategy.maxDepth - 1;
        strategy.useMissing = true;
        return strategy;
    }

    private static DataStream<Double> calcPrior(Table dataTable, BoostingStrategy strategy) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) dataTable).getTableEnvironment();
        DataStream<Row> data = tEnv.toDataStream(dataTable);
        if (null == strategy.baseScore) {
            LOG.info("Calculate prior by scanning data.");
            return DataStreamUtils.aggregate(data, new LabelSumCountFunction(strategy.labelCol))
                    .map(new CalcPriorFunction(strategy.lossType));
        } else {
            LOG.info("Use prior based on provided base score: {}", strategy.baseScore);
            double prior = baseScoreToPrior(strategy.baseScore, strategy.lossType);
            return tEnv.toDataStream(tEnv.fromValues(prior), Double.class);
        }
    }

    private static double baseScoreToPrior(double baseScore, LossType lossType) {
        switch (lossType) {
            case LOGISTIC:
                return new Logit().value(baseScore);
            case SQUARED:
                return baseScore;
            default:
                throw new UnsupportedOperationException("Unsupported loss.");
        }
    }

    private static class InitTrainContextFunction extends RichMapFunction<Integer, TrainContext> {
        private final String featureMetaBcName;
        private final String priorBcName;
        private final BoostingStrategy strategy;

        private InitTrainContextFunction(
                String featureMetaBcName, String priorBcName, BoostingStrategy strategy) {
            this.featureMetaBcName = featureMetaBcName;
            this.priorBcName = priorBcName;
            this.strategy = strategy;
        }

        @Override
        public TrainContext map(Integer value) {
            TrainContext trainContext = new TrainContext();
            trainContext.strategy = strategy;
            trainContext.featureMetas =
                    getRuntimeContext()
                            .<FeatureMeta>getBroadcastVariable(featureMetaBcName)
                            .toArray(new FeatureMeta[0]);
            if (!trainContext.strategy.isInputVector) {
                Arrays.sort(
                        trainContext.featureMetas,
                        Comparator.comparing(
                                d -> ArrayUtils.indexOf(strategy.featuresCols, d.name)));
            }
            trainContext.numFeatures = trainContext.featureMetas.length;
            trainContext.prior =
                    getRuntimeContext().<Double>getBroadcastVariable(priorBcName).get(0);
            return trainContext;
        }
    }

    private static class LabelSumCountFunction
            implements AggregateFunction<Row, Tuple2<Double, Long>, Tuple2<Double, Long>> {

        private final String labelCol;

        private LabelSumCountFunction(String labelCol) {
            this.labelCol = labelCol;
        }

        @Override
        public Tuple2<Double, Long> createAccumulator() {
            return Tuple2.of(0., 0L);
        }

        @Override
        public Tuple2<Double, Long> add(Row value, Tuple2<Double, Long> accumulator) {
            double label = ((Number) value.getFieldAs(labelCol)).doubleValue();
            return Tuple2.of(accumulator.f0 + label, accumulator.f1 + 1);
        }

        @Override
        public Tuple2<Double, Long> getResult(Tuple2<Double, Long> accumulator) {
            return accumulator;
        }

        @Override
        public Tuple2<Double, Long> merge(Tuple2<Double, Long> a, Tuple2<Double, Long> b) {
            return Tuple2.of(a.f0 + b.f0, a.f1 + b.f1);
        }
    }

    private static class CalcPriorFunction implements MapFunction<Tuple2<Double, Long>, Double> {
        private final LossType lossType;

        public CalcPriorFunction(LossType lossType) {
            this.lossType = lossType;
        }

        @Override
        public Double map(Tuple2<Double, Long> value) {
            switch (lossType) {
                case LOGISTIC:
                    return Math.log(value.f0 / (value.f1 - value.f0));
                case SQUARED:
                    return value.f0 / value.f1;
                default:
                    throw new UnsupportedOperationException("Unsupported loss.");
            }
        }
    }
}
