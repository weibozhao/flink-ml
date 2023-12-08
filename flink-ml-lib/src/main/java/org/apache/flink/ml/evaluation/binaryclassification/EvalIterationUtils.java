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

package org.apache.flink.ml.evaluation.binaryclassification;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.base.BooleanSerializer;
import org.apache.flink.api.common.typeutils.base.DoubleSerializer;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.typeutils.runtime.TupleSerializer;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.datastream.DataStreamInIterationUtils;
import org.apache.flink.ml.evaluation.binaryclassification.BinaryClassificationEvaluator.AppendTaskIdPureFunc;
import org.apache.flink.ml.evaluation.binaryclassification.BinaryClassificationEvaluator.BinarySummary;
import org.apache.flink.ml.evaluation.binaryclassification.BinaryClassificationEvaluator.CalcBoundaryRangeFunction;
import org.apache.flink.ml.evaluation.binaryclassification.BinaryClassificationEvaluator.ParseSample;
import org.apache.flink.ml.evaluation.binaryclassification.BinaryClassificationEvaluator.SampleScoreFunction;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static org.apache.flink.ml.evaluation.binaryclassification.BinaryClassificationEvaluator.updateBinarySummary;

/** This utility class provides methods to call eval functions in iterations. */
public class EvalIterationUtils {

    public static DataStream<Row> eval(
            DataStream<Row> data, String labelCol, String rawPredictionCol, String weightCol) {
        DataStream<Tuple3<Double, Boolean, Double>> evalData =
                data.map(new ParseSample(labelCol, rawPredictionCol, weightCol));

        DataStream<double[]> boundaryRange = getBoundaryRangeInIteration(evalData);

        //noinspection unchecked
        DataStream<Tuple4<Double, Boolean, Double, Integer>> evalDataWithTaskId =
                DataStreamInIterationUtils.mapWithBc(
                        evalData,
                        boundaryRange,
                        new AppendTaskIdPureFunc(),
                        new TupleSerializer<>(
                                (Class<Tuple3<Double, Boolean, Double>>) (Class<?>) Tuple3.class,
                                new TypeSerializer[] {
                                    DoubleSerializer.INSTANCE,
                                    BooleanSerializer.INSTANCE,
                                    DoubleSerializer.INSTANCE
                                }),
                        Types.TUPLE(Types.DOUBLE, Types.BOOLEAN, Types.DOUBLE, Types.INT));

        /* Repartition the evaluated data by range. */
        evalDataWithTaskId =
                evalDataWithTaskId.partitionCustom((chunkId, numPartitions) -> chunkId, x -> x.f3);

        /* Sorts local data by score.*/
        DataStream<Tuple3<Double, Boolean, Double>> sortEvalData =
                DataStreamInIterationUtils.mapPartition(
                        evalDataWithTaskId,
                        new MapPartitionFunction<
                                Tuple4<Double, Boolean, Double, Integer>,
                                Tuple3<Double, Boolean, Double>>() {
                            @Override
                            public void mapPartition(
                                    Iterable<Tuple4<Double, Boolean, Double, Integer>> values,
                                    Collector<Tuple3<Double, Boolean, Double>> out) {
                                List<Tuple3<Double, Boolean, Double>> bufferedData =
                                        new ArrayList<>();
                                for (Tuple4<Double, Boolean, Double, Integer> t4 : values) {
                                    bufferedData.add(Tuple3.of(t4.f0, t4.f1, t4.f2));
                                }
                                bufferedData.sort(Comparator.comparingDouble(o -> o.f0));
                                for (Tuple3<Double, Boolean, Double> dataPoint : bufferedData) {
                                    out.collect(dataPoint);
                                }
                            }
                        });
        sortEvalData.getTransformation().setName("sortEvalData");

        /* Calculates the summary of local data. */
        DataStream<BinarySummary> partitionSummaries =
                sortEvalData.transform(
                        "reduceInEachPartition",
                        TypeInformation.of(BinarySummary.class),
                        new PartitionSummaryInIterationOperator());

        /* Calculates intermediate results for each group of samples with same scores. */
        DataStream<Tuple3<Double, Double, Double>> partitionIntermediateResults =
                partitionSummaries
                        .broadcast()
                        .connect(sortEvalData)
                        .transform(
                                "CalcPartitionIntermediateResultsOperator",
                                Types.TUPLE(Types.DOUBLE, Types.DOUBLE, Types.DOUBLE),
                                new CalcPartitionIntermediateResultsInIterationOperator());

        DataStream<Tuple3<Double, Double, Double>> reducedIntermediateResults =
                DataStreamInIterationUtils.reduce(
                        partitionIntermediateResults,
                        (t1, t2) -> Tuple3.of(t1.f0 + t2.f0, t1.f1 + t2.f1, t1.f2 + t2.f2));

        DataStream<Double> areaUnderROC =
                reducedIntermediateResults.map(
                        t -> t.f1 > 0 && t.f2 > 0 ? 1 - t.f0 / t.f1 / t.f2 : Double.NaN);
        return areaUnderROC.map(Row::of);
    }

    static DataStream<double[]> getBoundaryRangeInIteration(
            DataStream<Tuple3<Double, Boolean, Double>> evalData) {
        DataStream<double[]> sampleScoreStream =
                DataStreamInIterationUtils.mapPartition(evalData, new SampleScoreFunction());
        sampleScoreStream.getTransformation().setName("sampleScore");
        final int parallel = sampleScoreStream.getParallelism();

        DataStream<double[]> boundaryRange =
                DataStreamInIterationUtils.mapPartition(
                        sampleScoreStream, new CalcBoundaryRangeFunction(parallel));
        boundaryRange.getTransformation().setName("boundaryRange");
        boundaryRange.getTransformation().setParallelism(1);
        return boundaryRange;
    }

    static class PartitionSummaryInIterationOperator extends AbstractStreamOperator<BinarySummary>
            implements OneInputStreamOperator<Tuple3<Double, Boolean, Double>, BinarySummary>,
                    IterationListener<Tuple3<Double, Boolean, Double>> {
        private ListState<BinarySummary> summaryState;
        private BinarySummary summary;

        @Override
        public void processElement(StreamRecord<Tuple3<Double, Boolean, Double>> streamRecord) {
            if (null == summary) {
                summary =
                        new BinarySummary(
                                getRuntimeContext().getIndexOfThisSubtask(),
                                -Double.MAX_VALUE,
                                0,
                                0);
            }
            updateBinarySummary(summary, streamRecord.getValue());
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            summaryState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "summaryState",
                                            TypeInformation.of(BinarySummary.class)));
            summary =
                    OperatorStateUtils.getUniqueElement(summaryState, "summaryState").orElse(null);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            summaryState.clear();
            if (summary != null) {
                summaryState.add(summary);
            }
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark,
                Context context,
                Collector<Tuple3<Double, Boolean, Double>> collector) {
            if (summary != null) {
                output.collect(new StreamRecord<>(summary));
                summary = null;
            }
        }

        @Override
        public void onIterationTerminated(
                Context context, Collector<Tuple3<Double, Boolean, Double>> collector) {}
    }

    static class CalcPartitionIntermediateResultsInIterationOperator
            extends AbstractStreamOperator<Tuple3<Double, Double, Double>>
            implements TwoInputStreamOperator<
                            BinarySummary,
                            Tuple3<Double, Boolean, Double>,
                            Tuple3<Double, Double, Double>>,
                    IterationListener<Tuple3<Double, Double, Double>> {

        private transient ListStateWithCache<BinarySummary> summariesState;
        private transient ListStateWithCache<Tuple3<Double, Boolean, Double>> samplesState;

        @Override
        public void processElement1(StreamRecord<BinarySummary> streamRecord) throws Exception {
            summariesState.add(streamRecord.getValue());
        }

        @Override
        public void processElement2(StreamRecord<Tuple3<Double, Boolean, Double>> streamRecord)
                throws Exception {
            samplesState.add(streamRecord.getValue());
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark,
                Context context,
                Collector<Tuple3<Double, Double, Double>> collector)
                throws Exception {
            // Sum of weights of positive samples of all groups before the current one (inclusive).
            double prefixSumWeightPos = 0.;
            List<BinarySummary> statistics = new ArrayList<>();
            summariesState.get().forEach(statistics::add);
            int subtaskId = getRuntimeContext().getIndexOfThisSubtask();
            for (BinarySummary stat : statistics) {
                if (stat.taskId < subtaskId) {
                    prefixSumWeightPos += stat.sumWeightsPos;
                }
            }
            summariesState.clear();

            // Sum of weights of positive samples in current group.
            double groupSumWeightPos = 0.;
            // Sum of weights of negative samples in current group.
            double groupSumWeightNeg = 0.;
            // Sum of weights of positive samples in this partition.
            double partitionSumWeightPos = 0.;
            // Sum of weights of negative samples in this partition.
            double partitionSumWeightNeg = 0.;

            // The score of last group.
            double lastScore = Double.NaN;
            // The `aggregated value` of current group.
            double aggregated = 0.;

            for (Tuple3<Double, Boolean, Double> value : samplesState.get()) {
                boolean isPos = value.f1;
                double score = value.f0;
                double weight = value.f2;

                if (lastScore != score) {
                    aggregated += (prefixSumWeightPos - groupSumWeightPos / 2) * groupSumWeightNeg;
                    lastScore = score;
                    groupSumWeightPos = 0.;
                    groupSumWeightNeg = 0.;
                }
                if (isPos) {
                    prefixSumWeightPos += weight;
                    groupSumWeightPos += weight;
                    partitionSumWeightPos += weight;
                } else {
                    groupSumWeightNeg += weight;
                    partitionSumWeightNeg += weight;
                }
            }
            samplesState.clear();
            if (Double.isNaN(lastScore)) {
                return;
            }
            aggregated += (prefixSumWeightPos - groupSumWeightPos / 2) * groupSumWeightNeg;
            collector.collect(Tuple3.of(aggregated, partitionSumWeightPos, partitionSumWeightNeg));
        }

        @Override
        public void onIterationTerminated(
                Context context, Collector<Tuple3<Double, Double, Double>> collector) {}

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            summariesState =
                    new ListStateWithCache<>(
                            Types.POJO(BinarySummary.class).createSerializer(new ExecutionConfig()),
                            getContainingTask(),
                            getRuntimeContext(),
                            context,
                            getOperatorID());
            //noinspection unchecked,rawtypes
            samplesState =
                    new ListStateWithCache<>(
                            (TypeSerializer<Tuple3<Double, Boolean, Double>>)
                                    (TypeSerializer)
                                            Types.TUPLE(Types.DOUBLE, Types.BOOLEAN, Types.DOUBLE)
                                                    .createSerializer(new ExecutionConfig()),
                            getContainingTask(),
                            getRuntimeContext(),
                            context,
                            getOperatorID());
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            summariesState.snapshotState(context);
            samplesState.snapshotState(context);
        }
    }
}
