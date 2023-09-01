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

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple5;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.ml.common.gbt.defs.BoostingStrategy;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.Split;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.gbt.operators.CacheDataCalcLocalHistsOperator;
import org.apache.flink.ml.common.gbt.operators.CalcLocalSplitsOperator;
import org.apache.flink.ml.common.gbt.operators.GetLabelPredOperator;
import org.apache.flink.ml.common.gbt.operators.PostSplitsOperator;
import org.apache.flink.ml.common.gbt.operators.PrintEvalResultsOperator;
import org.apache.flink.ml.common.gbt.operators.ReduceHistogramFunction;
import org.apache.flink.ml.common.gbt.operators.ReduceSplitsOperator;
import org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants;
import org.apache.flink.ml.common.gbt.operators.TerminationOperator;
import org.apache.flink.ml.common.sharedobjects.ItemDescriptor;
import org.apache.flink.ml.common.sharedobjects.SharedObjectsBody;
import org.apache.flink.ml.common.sharedobjects.SharedObjectsStreamOperator;
import org.apache.flink.ml.common.sharedobjects.SharedObjectsUtils;
import org.apache.flink.ml.evaluation.binaryclassification.EvalIterationUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.types.Row;
import org.apache.flink.util.OutputTag;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implements iteration body for boosting algorithms. This implementation uses horizontal partition
 * of data and row-store storage of instances.
 */
class BoostIterationBody implements IterationBody {
    private final BoostingStrategy strategy;

    public BoostIterationBody(BoostingStrategy strategy) {
        this.strategy = strategy;
    }

    private SharedObjectsBody.SharedObjectsBodyResult sharedObjectsBody(
            List<DataStream<?>> inputs) {
        //noinspection unchecked
        DataStream<Row> data = (DataStream<Row>) inputs.get(0);
        //noinspection unchecked
        DataStream<TrainContext> trainContext = (DataStream<TrainContext>) inputs.get(1);

        Map<ItemDescriptor<?>, SharedObjectsStreamOperator> ownerMap = new HashMap<>();

        CacheDataCalcLocalHistsOperator cacheDataCalcLocalHistsOp =
                new CacheDataCalcLocalHistsOperator(strategy);
        SingleOutputStreamOperator<Tuple5<Integer, Integer, Integer, Integer, Histogram>>
                localHists =
                        data.connect(trainContext)
                                .transform(
                                        "CacheDataCalcLocalHists",
                                        Types.TUPLE(
                                                Types.INT,
                                                Types.INT,
                                                Types.INT,
                                                Types.INT,
                                                TypeInformation.of(Histogram.class)),
                                        cacheDataCalcLocalHistsOp);
        for (ItemDescriptor<?> s : SharedObjectsConstants.OWNED_BY_CACHE_DATA_CALC_LOCAL_HISTS_OP) {
            ownerMap.put(s, cacheDataCalcLocalHistsOp);
        }

        DataStream<Tuple2<Integer, Histogram>> globalHists =
                localHists.keyBy(d -> d.f1).flatMap(new ReduceHistogramFunction());

        SingleOutputStreamOperator<Tuple3<Integer, Integer, Split>> localSplits =
                globalHists.transform(
                        "CalcLocalSplits",
                        Types.TUPLE(Types.INT, Types.INT, TypeInformation.of(Split.class)),
                        new CalcLocalSplitsOperator());

        DataStream<Tuple2<Integer, Split>> globalSplits =
                localSplits
                        .keyBy(d -> d.f0)
                        .transform(
                                "ReduceGlobalSplits",
                                Types.TUPLE(Types.INT, TypeInformation.of(Split.class)),
                                new ReduceSplitsOperator());

        PostSplitsOperator postSplitsOp = new PostSplitsOperator();
        SingleOutputStreamOperator<Integer> updatedModelData =
                globalSplits
                        .broadcast()
                        .transform("PostSplits", TypeInformation.of(Integer.class), postSplitsOp);
        for (ItemDescriptor<?> descriptor : SharedObjectsConstants.OWNED_BY_POST_SPLITS_OP) {
            ownerMap.put(descriptor, postSplitsOp);
        }

        // Evaluation with training data
        final String labelCol = "label";
        final String probCol = "prob";
        GetLabelPredOperator getLabelPredOp = new GetLabelPredOperator(labelCol, probCol, 10);
        DataStream<Row> trainLabelPred =
                updatedModelData.transform(
                        "GetTrainLabelPred",
                        Types.ROW_NAMED(
                                new String[] {labelCol, probCol}, Types.DOUBLE, Types.DOUBLE),
                        getLabelPredOp);
        DataStream<Row> trainEvalResults =
                EvalIterationUtils.eval(trainLabelPred, labelCol, probCol, null);
        trainEvalResults =
                trainEvalResults
                        .broadcast()
                        .transform(
                                "PrintTrainEvalResults",
                                trainEvalResults.getType(),
                                new PrintEvalResultsOperator());

        final OutputTag<GBTModelData> finalModelDataOutputTag =
                new OutputTag<>("model_data", TypeInformation.of(GBTModelData.class));
        SingleOutputStreamOperator<Integer> termination =
                updatedModelData.transform(
                        "CheckTermination",
                        Types.INT,
                        new TerminationOperator(finalModelDataOutputTag));
        DataStream<GBTModelData> finalModelData =
                termination.getSideOutput(finalModelDataOutputTag);

        return new SharedObjectsBody.SharedObjectsBodyResult(
                Arrays.asList(updatedModelData, finalModelData, termination, trainEvalResults),
                Arrays.asList(
                        localHists.getTransformation(),
                        localSplits.getTransformation(),
                        globalSplits.getTransformation(),
                        updatedModelData.getTransformation(),
                        termination.getTransformation(),
                        trainEvalResults.getTransformation()),
                ownerMap);
    }

    @Override
    public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
        DataStream<Row> data = dataStreams.get(0);
        DataStream<TrainContext> trainContext =
                variableStreams.<TrainContext>get(0).union(variableStreams.get(1));

        List<DataStream<?>> outputs =
                SharedObjectsUtils.withSharedObjects(
                        Arrays.asList(data, trainContext), this::sharedObjectsBody);

        DataStream<?> updatedModelData = outputs.get(0);
        DataStream<?> finalModelData = outputs.get(1);
        DataStream<?> termination = outputs.get(2);
        DataStream<?> trainEvalResults = outputs.get(3);

        return new IterationBodyResult(
                DataStreamList.of(
                        updatedModelData.flatMap(
                                (d, out) -> {}, TypeInformation.of(TrainContext.class)),
                        // add a feedback stream so that the epoch watermark is synchronized.
                        trainEvalResults.flatMap(
                                (d, out) -> {}, TypeInformation.of(TrainContext.class))),
                DataStreamList.of(finalModelData, trainEvalResults),
                termination);
    }
}
