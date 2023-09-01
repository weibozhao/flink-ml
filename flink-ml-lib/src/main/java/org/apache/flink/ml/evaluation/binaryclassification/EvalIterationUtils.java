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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
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
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.datastream.DataStreamInIterationUtils;
import org.apache.flink.ml.common.datastream.purefunc.RichMapWithBcPureFunc;
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
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

import static org.apache.flink.ml.evaluation.binaryclassification.BinaryClassificationEvaluator.reduceBinarySummary;
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
                                        new LinkedList<>();
                                for (Tuple4<Double, Boolean, Double, Integer> t4 : values) {
                                    bufferedData.add(Tuple3.of(t4.f0, t4.f1, t4.f2));
                                }
                                bufferedData.sort(Comparator.comparingDouble(o -> -o.f0));
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

        DataStream<List<BinarySummary>> partitionSummariesList =
                DataStreamInIterationUtils.mapPartition(
                        partitionSummaries,
                        (values, out) -> {
                            List<BinarySummary> lists = new ArrayList<>();
                            for (BinarySummary value : values) {
                                lists.add(value);
                            }
                            out.collect(lists);
                        },
                        Types.LIST(TypeInformation.of(BinarySummary.class)));
        partitionSummariesList.getTransformation().setParallelism(1);

        //noinspection unchecked
        DataStream<Tuple4<Double, Long, Boolean, Double>> dataWithOrders =
                DataStreamInIterationUtils.mapWithBc(
                        sortEvalData,
                        partitionSummariesList,
                        new CalcSampleOrdersPureFunc(),
                        new TupleSerializer<>(
                                (Class<Tuple3<Double, Boolean, Double>>) (Class<?>) Tuple3.class,
                                new TypeSerializer[] {
                                    DoubleSerializer.INSTANCE,
                                    BooleanSerializer.INSTANCE,
                                    DoubleSerializer.INSTANCE
                                }),
                        Types.TUPLE(Types.DOUBLE, Types.LONG, Types.BOOLEAN, Types.DOUBLE));

        DataStream<double[]> localAreaUnderROCVariable =
                dataWithOrders.transform(
                        "AccumulateMultiScore",
                        TypeInformation.of(double[].class),
                        new AccumulateMultiScoreInIterationOperator());

        DataStream<double[]> middleAreaUnderROC =
                DataStreamInIterationUtils.reduce(
                        localAreaUnderROCVariable,
                        (ReduceFunction<double[]>)
                                (t1, t2) -> {
                                    t2[0] += t1[0];
                                    t2[1] += t1[1];
                                    t2[2] += t1[2];
                                    return t2;
                                });

        DataStream<Double> areaUnderROC =
                middleAreaUnderROC.map(
                        (MapFunction<double[], Double>)
                                value -> {
                                    if (value[1] > 0 && value[2] > 0) {
                                        return (value[0] - 1. * value[1] * (value[1] + 1) / 2)
                                                / (value[1] * value[2]);
                                    } else {
                                        return Double.NaN;
                                    }
                                });
        return areaUnderROC.map(Row::of);

        //        Map<String, DataStream<?>> broadcastMap = new HashMap<>();
        //        broadcastMap.put(partitionSummariesKey, partitionSummaries);
        //        broadcastMap.put(AREA_UNDER_ROC, areaUnderROC);
        //        DataStream<BinaryClassificationEvaluator.BinaryMetrics> localMetrics =
        //                BroadcastUtils.withBroadcastStream(
        //                        Collections.singletonList(sortEvalData),
        //                        broadcastMap,
        //                        inputList -> {
        //                            DataStream input = inputList.get(0);
        //                            DataStream dataStream =
        //                                    DataStreamUtils.mapPartition(
        //                                            input, new
        // CalcBinaryMetrics(partitionSummariesKey));
        //                            return dataStream;
        //                        });
        //
        //        DataStream<Map<String, Double>> metrics =
        //                DataStreamUtils.mapPartition(
        //                        localMetrics, new MergeMetrics(), Types.MAP(Types.STRING,
        // Types.DOUBLE));
        //        metrics.getTransformation().setParallelism(1);
        //
        //        TypeInformation<?>[] metricTypes = new TypeInformation[metricsNames.length];
        //        Arrays.fill(metricTypes, Types.DOUBLE);
        //        RowTypeInfo outputTypeInfo = new RowTypeInfo(metricTypes, metricsNames);
        //
        //        DataStream<Row> evalResult =
        //                metrics.map(
        //                        (MapFunction<Map<String, Double>, Row>)
        //                                value -> {
        //                                    Row ret = new Row(metricsNames.length);
        //                                    for (int i = 0; i < metricsNames.length; ++i) {
        //                                        ret.setField(i, value.get(metricsNames[i]));
        //                                    }
        //                                    return ret;
        //                                },
        //                        outputTypeInfo);
        //        return evalResult;
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
                    OperatorStateUtils.getUniqueElement(summaryState, "summaryState")
                            .orElse(
                                    new BinarySummary(
                                            getRuntimeContext().getIndexOfThisSubtask(),
                                            -Double.MAX_VALUE,
                                            0,
                                            0));
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

    static class CalcSampleOrdersPureFunc
            extends RichMapWithBcPureFunc<
                    Tuple3<Double, Boolean, Double>,
                    Tuple4<Double, Long, Boolean, Double>,
                    List<BinarySummary>> {

        private long[] countValues;
        private long startIndex;
        private long total;

        @Override
        public void open() {
            countValues = null;
        }

        @Override
        public Tuple4<Double, Long, Boolean, Double> map(
                Tuple3<Double, Boolean, Double> value, List<BinarySummary> statistics) {
            if (null == countValues) {
                countValues = reduceBinarySummary(statistics, getContext().getSubtaskId());
                startIndex = countValues[1] + countValues[0] + 1;
                total = countValues[2] + countValues[3];
            }
            Tuple4<Double, Long, Boolean, Double> out =
                    Tuple4.of(value.f0, total - startIndex + 1, value.f1, value.f2);
            startIndex++;
            return out;
        }
    }

    static class AccumulateMultiScoreInIterationOperator extends AbstractStreamOperator<double[]>
            implements OneInputStreamOperator<Tuple4<Double, Long, Boolean, Double>, double[]>,
                    IterationListener<double[]> {
        double[] accValue;
        double score;
        private ListState<double[]> accValueState;
        private ListState<Double> scoreState;

        @Override
        public void processElement(
                StreamRecord<Tuple4<Double, Long, Boolean, Double>> streamRecord) {
            Tuple4<Double, Long, Boolean, Double> t = streamRecord.getValue();
            if (accValue == null) {
                accValue = new double[4];
                score = t.f0;
            } else if (score != t.f0) {
                output.collect(
                        new StreamRecord<>(
                                new double[] {
                                    accValue[0] / accValue[1] * accValue[2],
                                    accValue[2],
                                    accValue[3]
                                }));
                Arrays.fill(accValue, 0.0);
            }
            accValue[0] += t.f1;
            accValue[1] += 1.0;
            if (t.f2) {
                accValue[2] += t.f3;
            } else {
                accValue[3] += t.f3;
            }
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            accValueState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "accValueState", TypeInformation.of(double[].class)));
            accValue =
                    OperatorStateUtils.getUniqueElement(accValueState, "accValueState")
                            .orElse(null);

            scoreState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "scoreState", TypeInformation.of(Double.class)));
            score = OperatorStateUtils.getUniqueElement(scoreState, "scoreState").orElse(0.0);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            accValueState.clear();
            scoreState.clear();
            if (accValue != null) {
                accValueState.add(accValue);
                scoreState.add(score);
            }
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<double[]> collector) {
            if (accValue != null) {
                output.collect(
                        new StreamRecord<>(
                                new double[] {
                                    accValue[0] / accValue[1] * accValue[2],
                                    accValue[2],
                                    accValue[3]
                                }));
                accValue = null;
                score = 0.;
            }
        }

        @Override
        public void onIterationTerminated(Context context, Collector<double[]> collector) {}
    }
}
