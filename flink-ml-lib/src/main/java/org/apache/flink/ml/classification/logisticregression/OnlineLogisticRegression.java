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

package org.apache.flink.ml.classification.logisticregression;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.ObjectArrayTypeInfo;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.ps.api.AlgorithmFlow;
import org.apache.flink.ml.common.ps.api.CoTransformComponent;
import org.apache.flink.ml.common.ps.api.MLData;
import org.apache.flink.ml.common.ps.api.MLData.MLDataFunction;
import org.apache.flink.ml.common.ps.api.MiniBatchComponent;
import org.apache.flink.ml.common.ps.api.TransformComponent;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An Estimator which implements the online logistic regression algorithm. The online optimizer of
 * this algorithm is The FTRL-Proximal proposed by H.Brendan McMahan et al.
 *
 * <p>See <a href="https://doi.org/10.1145/2487575.2488200">H. Brendan McMahan et al., Ad click
 * prediction: a view from the trenches.</a>
 */
public class OnlineLogisticRegression
        implements Estimator<OnlineLogisticRegression, OnlineLogisticRegressionModel>,
                OnlineLogisticRegressionParams<OnlineLogisticRegression> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table initModelDataTable;

    public OnlineLogisticRegression() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public OnlineLogisticRegressionModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        MLData mlData =
                new MLData(
                        new Table[] {inputs[0], initModelDataTable},
                        new String[] {"data", "initModel"});

        AlgorithmFlow flow =
                new AlgorithmFlow(false)
                        .add(
                                new MLDataFunction(
                                        "map",
                                        new FeaturesLabelExtractor(
                                                getFeaturesCol(), getLabelCol(), getWeightCol())))
                        .add(
                                new MLDataFunction(
                                                "map",
                                                (MapFunction<Row, DenseVector>)
                                                        x ->
                                                                (new LogisticRegressionModelData(
                                                                                x.getFieldAs(0),
                                                                                x.getFieldAs(1)))
                                                                        .coefficient)
                                        .withParallel(1)
                                        .input("initModel")
                                        .output("initModel"))
                        .startIteration(new String[] {"initModel"}, new String[] {"data"}, false)
                        .add(new MiniBatchComponent(getGlobalBatchSize()))
                        .add(new MLDataFunction("broadcast").input("initModel").output("initModel"))
                        .add(
                                new CalculateLocalGradient()
                                        .input("data")
                                        .with("initModel")
                                        .output("grad")
                                        .returns(TypeInformation.of(DenseVector[].class)))
                        .add(
                                new MLDataFunction(
                                                "reduce",
                                                new ReduceFunction<DenseVector[]>() {
                                                    @Override
                                                    public DenseVector[] reduce(
                                                            DenseVector[] gradientInfo,
                                                            DenseVector[] newGradientInfo) {
                                                        BLAS.axpy(
                                                                1.0,
                                                                gradientInfo[0],
                                                                newGradientInfo[0]);
                                                        BLAS.axpy(
                                                                1.0,
                                                                gradientInfo[1],
                                                                newGradientInfo[1]);
                                                        if (newGradientInfo[2] == null) {
                                                            newGradientInfo[2] = gradientInfo[2];
                                                        }
                                                        return newGradientInfo;
                                                    }
                                                })
                                        .isOnine(true))
                        .add(
                                new UpdateModel(
                                                getAlpha(),
                                                getBeta(),
                                                getReg() * getElasticNet(),
                                                (1 - getElasticNet()) * getReg())
                                        .withParallel(1)
                                        .returns(DenseVectorTypeInfo.INSTANCE))
                        .add(
                                new MLDataFunction("map", new CreateLrModelData())
                                        .withParallel(1)
                                        .input("grad")
                                        .output("outModel"))
                        .endIteration(new String[] {"grad", "outModel"}, false);

        Table onlineModelDataTable = flow.apply(mlData).getTable();
        OnlineLogisticRegressionModel model =
                new OnlineLogisticRegressionModel().setModelData(onlineModelDataTable);
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    private static class FeaturesLabelExtractor implements MapFunction<Row, Row> {
        private final String featuresCol;
        private final String labelCol;
        private final String weightCol;

        private FeaturesLabelExtractor(String featuresCol, String labelCol, String weightCol) {
            this.featuresCol = featuresCol;
            this.labelCol = labelCol;
            this.weightCol = weightCol;
        }

        @Override
        public Row map(Row row) throws Exception {
            if (weightCol == null) {
                return Row.of(row.getField(featuresCol), row.getField(labelCol));
            } else {
                return Row.of(
                        row.getField(featuresCol), row.getField(labelCol), row.getField(weightCol));
            }
        }
    }

    private static class CreateLrModelData
            implements MapFunction<DenseVector, LogisticRegressionModelData>, CheckpointedFunction {
        private Long modelVersion = 1L;
        protected transient ListState modelVersionState;

        @Override
        public LogisticRegressionModelData map(DenseVector denseVector) throws Exception {
            return new LogisticRegressionModelData(denseVector, modelVersion++);
        }

        @Override
        public void snapshotState(FunctionSnapshotContext functionSnapshotContext)
                throws Exception {
            modelVersionState.update(Collections.singletonList(modelVersion));
        }

        @Override
        public void initializeState(FunctionInitializationContext context) throws Exception {
            modelVersionState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>("modelVersionState", Long.class));
        }
    }

    /** Updates model. */
    private static class UpdateModel extends TransformComponent<DenseVector[], DenseVector> {
        private ListState<double[]> nParamState;
        private ListState<double[]> zParamState;
        private final double alpha;
        private final double beta;
        private final double l1;
        private final double l2;
        private double[] nParam;
        private double[] zParam;

        public UpdateModel(double alpha, double beta, double l1, double l2) {
            this.alpha = alpha;
            this.beta = beta;
            this.l1 = l1;
            this.l2 = l2;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            nParamState =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("nParamState", double[].class));
            zParamState =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("zParamState", double[].class));
        }

        @Override
        public void processElement(StreamRecord<DenseVector[]> streamRecord) throws Exception {
            DenseVector[] gradientInfo = streamRecord.getValue();
            double[] coefficient = gradientInfo[2].values;
            double[] g = gradientInfo[0].values;
            for (int i = 0; i < g.length; ++i) {
                if (gradientInfo[1].values[i] != 0.0) {
                    g[i] = g[i] / gradientInfo[1].values[i];
                }
            }
            if (zParam == null) {
                zParam = new double[g.length];
                nParam = new double[g.length];
                nParamState.add(nParam);
                zParamState.add(zParam);
            }

            for (int i = 0; i < zParam.length; ++i) {
                double sigma = (Math.sqrt(nParam[i] + g[i] * g[i]) - Math.sqrt(nParam[i])) / alpha;
                zParam[i] += g[i] - sigma * coefficient[i];
                nParam[i] += g[i] * g[i];

                if (Math.abs(zParam[i]) <= l1) {
                    coefficient[i] = 0.0;
                } else {
                    coefficient[i] =
                            ((zParam[i] < 0 ? -1 : 1) * l1 - zParam[i])
                                    / ((beta + Math.sqrt(nParam[i])) / alpha + l2);
                }
            }
            output.collect(new StreamRecord<>(new DenseVector(coefficient)));
        }
    }

    private static class CalculateLocalGradient
            extends CoTransformComponent<Row[], DenseVector, DenseVector[]> {
        private ListState<DenseVector> modelDataState;
        private ListState<Row[]> localBatchDataState;
        private double[] gradient;
        private double[] weightSum;

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            modelDataState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>("modelData", DenseVector.class));
            TypeInformation<Row[]> type =
                    ObjectArrayTypeInfo.getInfoFor(TypeInformation.of(Row.class));
            localBatchDataState =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("localBatch", type));
        }

        @Override
        public void processElement1(StreamRecord<Row[]> pointsRecord) throws Exception {
            localBatchDataState.add(pointsRecord.getValue());
            calculateGradient();
        }

        @Override
        public void processElement2(StreamRecord<DenseVector> modelDataRecord) throws Exception {
            modelDataState.add(modelDataRecord.getValue());
            calculateGradient();
        }

        private void calculateGradient() throws Exception {
            if (!modelDataState.get().iterator().hasNext()
                    || !localBatchDataState.get().iterator().hasNext()) {
                return;
            }

            DenseVector modelData =
                    OperatorStateUtils.getUniqueElement(modelDataState, "modelData").get();
            modelDataState.clear();

            List<Row[]> pointsList = IteratorUtils.toList(localBatchDataState.get().iterator());
            Row[] points = pointsList.remove(0);
            localBatchDataState.update(pointsList);

            for (Row point : points) {
                Vector vec = point.getFieldAs(0);
                double label = point.getFieldAs(1);
                double weight = point.getArity() == 2 ? 1.0 : point.getFieldAs(2);
                if (gradient == null) {
                    gradient = new double[vec.size()];
                    weightSum = new double[gradient.length];
                }
                double p = BLAS.dot(modelData, vec);
                p = 1 / (1 + Math.exp(-p));
                if (vec instanceof DenseVector) {
                    DenseVector denseVector = (DenseVector) vec;
                    for (int i = 0; i < modelData.size(); ++i) {
                        gradient[i] += (p - label) * denseVector.values[i];
                        weightSum[i] += 1.0;
                    }
                } else {
                    SparseVector sparseVector = (SparseVector) vec;
                    for (int i = 0; i < sparseVector.indices.length; ++i) {
                        int idx = sparseVector.indices[i];
                        gradient[idx] += (p - label) * sparseVector.values[i];
                        weightSum[idx] += weight;
                    }
                }
            }

            if (points.length > 0) {
                output.collect(
                        new StreamRecord<>(
                                new DenseVector[] {
                                    new DenseVector(gradient),
                                    new DenseVector(weightSum),
                                    (getRuntimeContext().getIndexOfThisSubtask() == 0)
                                            ? modelData
                                            : null
                                }));
            }
            Arrays.fill(gradient, 0.0);
            Arrays.fill(weightSum, 0.0);
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                LogisticRegressionModelDataUtil.getModelDataStream(initModelDataTable),
                path,
                new LogisticRegressionModelDataUtil.ModelDataEncoder());
    }

    public static OnlineLogisticRegression load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        OnlineLogisticRegression onlineLogisticRegression = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new LogisticRegressionModelDataUtil.ModelDataDecoder());
        onlineLogisticRegression.setInitialModelData(modelDataTable);
        return onlineLogisticRegression;
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * Sets the initial model data of the online training process with the provided model data
     * table.
     */
    public OnlineLogisticRegression setInitialModelData(Table initModelDataTable) {
        this.initModelDataTable = initModelDataTable;
        return this;
    }
}
