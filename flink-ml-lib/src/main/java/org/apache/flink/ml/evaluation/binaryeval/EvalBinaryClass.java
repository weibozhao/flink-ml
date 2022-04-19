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

package org.apache.flink.ml.evaluation.binaryeval;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.EndOfStreamWindows;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.functions.windowing.WindowFunction;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.StreamMap;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Calculates the evaluation metrics for binary classification.
 */
public class EvalBinaryClass
        implements Transformer<EvalBinaryClass>, EvalBinaryClassParams<EvalBinaryClass> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private static final int NUM_SAMPLE = 100;
    public static int TOTAL_TRUE = 2;
    public static int TOTAL_FALSE = 3;
    public static int CUR_TRUE = 0;
    public static int CUR_FALSE = 1;
    public static int RECORD_LEN = 4;
    private static final double PROBABILITY_ERROR = 0.001;

    public EvalBinaryClass() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Tuple3<Double, Boolean, Double>>
            sampleStatistics =
            DataStreamUtils.mapPartition(
                tEnv.toDataStream(inputs[0]),
                new RichMapPartitionFunction<
                    Row,
                    Tuple3<Double, Boolean, Double>>() {
                    @Override
                    public void mapPartition(
                        Iterable<Row> values,
                        Collector<Tuple3<Double, Boolean, Double>> out) {
                        for (Row value : values) {
                            double label = value.getFieldAs(getLabelCol());
                            DenseVector probVec =
                                value.getFieldAs(getRawPredictionCol());
                            double prob = probVec.get(label == 1.0 ? 1 : 0);
                            double logLoss =
                                -Math.log(
                                    Math.max(Math.min(prob, 1 - 1e-15), 1e-15));
                            out.collect(Tuple3.of(probVec.get(1), label == 1.0, logLoss));
                        }
                        if (getRuntimeContext().getIndexOfThisSubtask() == 0) {
                            out.collect(Tuple3.of(0.5, true, Double.NaN));
                        }
                    }
                });

        DataStream<Tuple4<Double, Boolean, Double, Integer>> sampleStatisticsWithTaskId =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(sampleStatistics),
                        Collections.singletonMap("rangeBoundary", getBoundaryRange(sampleStatistics)),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return input.map(new AppendTaskId());
                        });

        sampleStatisticsWithTaskId =
                sampleStatisticsWithTaskId
                        .partitionCustom((chunkId, numPartitions) -> chunkId, x -> x.f3);

        sampleStatistics =
                DataStreamUtils.mapPartition(
                    sampleStatisticsWithTaskId,
                        new RichMapPartitionFunction<
                            Tuple4<Double, Boolean, Double, Integer>,
                                Tuple3<Double, Boolean, Double>>() {
                            @Override
                            public void mapPartition(
                                    Iterable<Tuple4<Double, Boolean, Double, Integer>> values,
                                    Collector<Tuple3<Double, Boolean, Double>> out) {
                                List<Tuple3<Double, Boolean, Double>> list = new ArrayList<>();
                                for (Tuple4<Double, Boolean, Double, Integer> t4 : values) {
                                    list.add(Tuple3.of(t4.f0, t4.f1, t4.f2));
                                }
                                list.sort(Comparator.comparingDouble(o -> -o.f0));
                                for (Tuple3<Double, Boolean, Double> t3 : list) {
                                    out.collect(t3);
                                }
                            }
                        });

        DataStream<BinaryPartitionSummary> partitionSummaries =
                DataStreamUtils.mapPartition(
                        sampleStatistics,
                        new RichMapPartitionFunction<
                                Tuple3<Double, Boolean, Double>, BinaryPartitionSummary>() {
                            @Override
                            public void mapPartition(
                                    Iterable<Tuple3<Double, Boolean, Double>> values,
                                    Collector<BinaryPartitionSummary> out) {
                                BinaryPartitionSummary statistics =
                                        new BinaryPartitionSummary(
                                                getRuntimeContext().getIndexOfThisSubtask(),
                                                -Double.MAX_VALUE,
                                                0,
                                                0);
                                values.forEach(
                                        t -> updateBinaryPartitionSummary(statistics, t, 0.5));
                                out.collect(statistics);
                            }
                        });

        DataStream<Tuple3<Double, Long, Boolean>> sampleOrders =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(sampleStatistics),
                        Collections.singletonMap("partitionSummaries", partitionSummaries),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return input.flatMap(new CalcSampleOrders());
                        });

        sampleOrders = sampleOrders.transform("appendMaxWaterMark", sampleOrders.getType(), new AppendMaxWatermark(x -> x));
        DataStream<Double> localAuc =
                sampleOrders

            .keyBy(new KeySelector <Tuple3<Double, Long, Boolean>, Double>() {
            @Override
            public Double getKey(Tuple3 <Double, Long, Boolean> value) throws Exception {
                return value.f0;
            }
         }).window(EndOfStreamWindows.get())
            .apply(new WindowFunction <Tuple3 <Double, Long, Boolean>, Double, Double,
                            TimeWindow>() {
                @Override
                public void apply(Double key, TimeWindow window, Iterable <Tuple3 <Double, Long,
         Boolean>> values,
                                  Collector <Double> out) throws Exception {
                        long sum = 0;
                        long cnt = 0;
                        long positiveCnt = 0;
                        for (Tuple3 <Double, Long, Boolean> t : values) {
                            System.out.println(t);
                            sum += t.f1;
                            cnt++;
                            if (t.f2) {
                                positiveCnt++;
                            }
                        }
                        System.out.println(1. * sum / cnt * positiveCnt);
                        out.collect(1. * sum / cnt * positiveCnt);

                }
            });
        DataStream<Double> auc =
                DataStreamUtils.mapPartition(
                        localAuc,
                        new RichMapPartitionFunction<Double, Double>() {
                            @Override
                            public void mapPartition(Iterable<Double> values, Collector<Double> out) {
                                double sum = 0.0;
                                for (double value : values) {
                                    sum += value;
                                }
                                System.out.println(sum);
                                out.collect(sum);
                            }
                        });
        auc.getTransformation().setParallelism(1);

        Map<String, DataStream<?>> broadcastMap = new HashMap<>();
        broadcastMap.put("partitionSummaries", partitionSummaries);
        broadcastMap.put("auc", auc);
        DataStream<BinaryMetricsSummary> summary =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(sampleStatistics),
                        broadcastMap,
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return DataStreamUtils.mapPartition(
                                    input, new CalcBinaryMetricsSummary()); // todo check correct
                        });

        DataStream<BinaryMetricsSummary> metrics =
                DataStreamUtils.mapPartition(summary, new MergeMetrics());
        metrics.getTransformation().setParallelism(1);

        return new Table[] {};
    }
    /** Merge the BaseMetrics calculated locally. */
    public static class MergeMetrics
            extends RichMapPartitionFunction<BinaryMetricsSummary, BinaryMetricsSummary> {
        private static final long serialVersionUID = 463407033215369847L;

        @Override
        public void mapPartition(
                Iterable<BinaryMetricsSummary> values, Collector<BinaryMetricsSummary> out) {
            BinaryMetricsSummary ret = null;
            for (BinaryMetricsSummary summary : values) {
                ret = (ret == null) ? summary : ret.merge(summary);
            }
            System.out.println(ret.confusionMatrix);
            System.out.println("acc: " + BinaryMetricsSummary.accuracy(ret.confusionMatrix));
            System.out.println("kapa: " + BinaryMetricsSummary.kappa(ret.confusionMatrix));
            System.out.println("f1: " + BinaryMetricsSummary.f1(ret.confusionMatrix));
            System.out.println("prec: " + BinaryMetricsSummary.precision(ret.confusionMatrix));

            System.out.println("auc: " + ret.auc);
            System.out.println("ks: " + ret.ks);
            System.out.println("gini: " + ret.gini);
            System.out.println("logloss: " + ret.logLoss);
            System.out.println("prc: " + ret.prc);
            System.out.println("accuracy: " + ret.accuracy);
            out.collect(ret);
        }
    }

    static class CalcBinaryMetricsSummary
            extends RichMapPartitionFunction<
                    Tuple3<Double, Boolean, Double>, BinaryMetricsSummary> {
        private static final long serialVersionUID = 5680342197308160013L;
        private long[] countValues;
        private boolean firstBin;
        private double auc;
        private double decisionThreshold;

        @Override
        public void mapPartition(
                Iterable<Tuple3<Double, Boolean, Double>> iterable,
                Collector<BinaryMetricsSummary> collector) {

            if (true) {
                List<BinaryPartitionSummary> statistics =
                        getRuntimeContext().getBroadcastVariable("partitionSummaries");
                Tuple2<Boolean, long[]> t =
                        reduceBinaryPartitionSummary(
                                statistics, getRuntimeContext().getIndexOfThisSubtask());
                firstBin = t.f0;
                countValues = t.f1;

                auc = getRuntimeContext().<Double>getBroadcastVariable("auc").get(0);
                long totalTrue = countValues[TOTAL_TRUE];
                long totalFalse = countValues[TOTAL_FALSE];
                if (totalTrue == 0) {
                    System.out.println("There is no positive sample in data!");
                }
                if (totalFalse == 0) {
                    System.out.println("There is no negative sample in data!");
                }
                if (totalTrue > 0 && totalFalse > 0) {
                    auc = (auc - 1. * totalTrue * (totalTrue + 1) / 2) / (totalTrue * totalFalse);
                } else {
                    auc = Double.NaN;
                }

                decisionThreshold = 0.5;
            }
            BinaryMetricsSummary summary =
                    new BinaryMetricsSummary(decisionThreshold, 0.0, 0L, auc);
            double[] tprFprPrecision = new double[RECORD_LEN];
            for (Tuple3<Double, Boolean, Double> t : iterable) {
                updateBinaryMetricsSummary(
                        t,
                        summary,
                        countValues,
                        tprFprPrecision,
                        firstBin,
                        decisionThreshold);
            }
            collector.collect(summary);
        }
    }

    public static void updateBinaryMetricsSummary(
            Tuple3<Double, Boolean, Double> cur,
            BinaryMetricsSummary binaryMetricsSummary,
            long[] countValues,
            double[] recordValues,
            boolean first,
            double decisionThreshold) {
        int POSITIVE_RATE = 3;
        int PRECISION = 2;
        int FPR = 1;
        int TPR = 0;
        if (binaryMetricsSummary.total == 0) {
            recordValues[TPR] =
                    countValues[TOTAL_TRUE] == 0
                            ? 1.0
                            : 1.0 * countValues[CUR_TRUE] / countValues[TOTAL_TRUE];
            recordValues[FPR] =
                    countValues[TOTAL_FALSE] == 0
                            ? 1.0
                            : 1.0 * countValues[CUR_FALSE] / countValues[TOTAL_FALSE];
            recordValues[PRECISION] =
                    countValues[CUR_TRUE] + countValues[CUR_FALSE] == 0
                            ? 1.0
                            : 1.0
                                    * countValues[CUR_TRUE]
                                    / (countValues[CUR_TRUE] + countValues[CUR_FALSE]);
            recordValues[POSITIVE_RATE] =
                    1.0
                            * (countValues[CUR_TRUE] + countValues[CUR_FALSE])
                            / (countValues[TOTAL_TRUE] + countValues[TOTAL_FALSE]);
        }

        if (!isMiddlePoint(cur, decisionThreshold)) {
            binaryMetricsSummary.total++;
            binaryMetricsSummary.logLoss += cur.f2;
            if (cur.f1) {
                countValues[CUR_TRUE]++;
            } else {
                countValues[CUR_FALSE]++;
            }
        }

        double threshold = cur.f0;
        double tpr =
                countValues[TOTAL_TRUE] == 0
                        ? 1.0
                        : 1.0 * countValues[CUR_TRUE] / countValues[TOTAL_TRUE];
        double fpr =
                countValues[TOTAL_FALSE] == 0
                        ? 1.0
                        : 1.0 * countValues[CUR_FALSE] / countValues[TOTAL_FALSE];
        double precision =
                countValues[CUR_TRUE] + countValues[CUR_FALSE] == 0
                        ? 1.0
                        : 1.0
                                * countValues[CUR_TRUE]
                                / (countValues[CUR_TRUE] + countValues[CUR_FALSE]);
        double positiveRate =
                1.0
                        * (countValues[CUR_TRUE] + countValues[CUR_FALSE])
                        / (countValues[TOTAL_TRUE] + countValues[TOTAL_FALSE]);

        if (binaryMetricsSummary.total == 1 && first) {
            recordValues[PRECISION] = precision;
            ConfusionMatrix confusionMatrix =
                    new ConfusionMatrix(
                            new long[][] {
                                {0, 0}, {countValues[TOTAL_TRUE], countValues[TOTAL_FALSE]}
                            });
            binaryMetricsSummary.confusionMatrix = confusionMatrix;
        }

        binaryMetricsSummary.gini +=
                ((positiveRate - recordValues[POSITIVE_RATE]) * (tpr + recordValues[TPR]) / 2);
        binaryMetricsSummary.prc +=
                ((tpr - recordValues[TPR]) * (precision + recordValues[PRECISION]) / 2);
        binaryMetricsSummary.ks = Math.max(Math.abs(fpr - tpr), binaryMetricsSummary.ks);

        recordValues[TPR] = tpr;
        recordValues[FPR] = fpr;
        recordValues[PRECISION] = precision;
        recordValues[POSITIVE_RATE] = positiveRate;

        ConfusionMatrix confusionMatrix =
                new ConfusionMatrix(
                        new long[][] {
                            {countValues[CUR_TRUE], countValues[CUR_FALSE]},
                            {
                                countValues[TOTAL_TRUE] - countValues[CUR_TRUE],
                                countValues[TOTAL_FALSE] - countValues[CUR_FALSE]
                            }
                        });

        // keep the middlePoint(p = decisionThreshold), keep the first point(p = 1.0), then compare
        // the threshold
        if (binaryMetricsSummary.confusionMatrix == null
                && (isMiddlePoint(cur, decisionThreshold))) {
            binaryMetricsSummary.confusionMatrix = confusionMatrix;
        }
    }

    /**
     * For each sample, calculate its score order among all samples. The sample with minimum score
     * has order 1, while the sample with maximum score has order #samples.
     *
     * <p>Input is a dataset of tuple (score, is real positive, logloss), output is a dataset of
     * tuple (score, order, is real positive).
     */
    static class CalcSampleOrders
            extends RichFlatMapFunction<
                    Tuple3<Double, Boolean, Double>, Tuple3<Double, Long, Boolean>> {
        private static final long serialVersionUID = 3047511137846831576L;
        private long startIndex;
        private long total = -1;

        @Override
        public void flatMap(
                Tuple3<Double, Boolean, Double> value, Collector<Tuple3<Double, Long, Boolean>> out)
                throws Exception {
            if (total == -1) {
                List<BinaryPartitionSummary> statistics =
                        getRuntimeContext().getBroadcastVariable("partitionSummaries");
                Tuple2<Boolean, long[]> t =
                        reduceBinaryPartitionSummary(
                                statistics, getRuntimeContext().getIndexOfThisSubtask());
                startIndex = t.f1[1] + t.f1[0] + 1;
                total = t.f1[2] + t.f1[3];
            }
            if (!isMiddlePoint(value, 0.5)) {
                out.collect(Tuple3.of(value.f0, total - startIndex + 1, value.f1));
                startIndex++;
            }
        }
    }
    /**
     * @param values Summary of different partitions.
     * @param taskId current taskId.
     * @return <The first partition, [curTrue, curFalse, TotalTrue, TotalFalse])
     */
    public static Tuple2<Boolean, long[]> reduceBinaryPartitionSummary(
            List<BinaryPartitionSummary> values, int taskId) {
        List<BinaryPartitionSummary> list = new ArrayList<>(values);
        list.sort(Comparator.comparingDouble(t -> -t.maxScore));
        long curTrue = 0;
        long curFalse = 0;
        long totalTrue = 0;
        long totalFalse = 0;

        boolean firstBin = true;

        for (BinaryPartitionSummary statistics : list) {
            if (statistics.taskId == taskId) {
                firstBin = (totalTrue + totalFalse == 0);
                curFalse = totalFalse;
                curTrue = totalTrue;
            }
            totalTrue += statistics.curPositive;
            totalFalse += statistics.curNegative;
        }
        return Tuple2.of(firstBin, new long[] {curTrue, curFalse, totalTrue, totalFalse});
    }

    public static void updateBinaryPartitionSummary(
            BinaryPartitionSummary statistics,
            Tuple3<Double, Boolean, Double> t,
            double middleThreshold) {
        if (!isMiddlePoint(t, middleThreshold)) {
            if (t.f1) {
                statistics.curPositive++;
            } else {
                statistics.curNegative++;
            }
        }
        int compare = Double.compare(statistics.maxScore, t.f0);
        if (compare < 0) {
            statistics.maxScore = t.f0;
        }
    }

    public static boolean isMiddlePoint(Tuple3<Double, Boolean, Double> t, double middleThreshold) {
        return Double.compare(t.f0, middleThreshold) == 0 && t.f1 && Double.isNaN(t.f2);
    }

    private static class AppendTaskId
            extends RichMapFunction<
                    Tuple3<Double, Boolean, Double>, Tuple4<Double, Boolean, Double, Integer>> {
        private double[] rangeBoundary;

        @Override
        public Tuple4<Double, Boolean, Double, Integer> map(Tuple3<Double, Boolean, Double> value)
                throws Exception {
            if (rangeBoundary == null) {
                rangeBoundary =
                        (double[]) getRuntimeContext().getBroadcastVariable("rangeBoundary").get(0);
            }
            for (int i = rangeBoundary.length - 1; i > 0; --i) {
                if (value.f0 > rangeBoundary[i]) {
                    return Tuple4.of(value.f0, value.f1, value.f2, i);
                }
            }
            return Tuple4.of(value.f0, value.f1, value.f2, 0);
        }
    }

    public static class BinaryPartitionSummary implements Serializable {
        private static final long serialVersionUID = 1L;
        Integer taskId;
        // maximum score in this partition
        double maxScore;
        // #real positives in this partition
        long curPositive;
        // #real negatives in this partition
        long curNegative;

        public BinaryPartitionSummary(
                Integer taskId, double maxScore, long curPositive, long curNegative) {
            this.taskId = taskId;
            this.maxScore = maxScore;
            this.curPositive = curPositive;
            this.curNegative = curNegative;
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static EvalBinaryClass load(StreamTableEnvironment env, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    private static class AppendMaxWatermark<IN, OUT> extends StreamMap <IN, OUT> implements BoundedOneInput {

        public AppendMaxWatermark(MapFunction <IN, OUT> mapper) {
            super(mapper);
        }

        @Override
        public void endInput() throws Exception {
            output.emitWatermark(Watermark.MAX_WATERMARK);
        }
    }

    private static DataStream<double[]> getBoundaryRange(DataStream<Tuple3<Double, Boolean, Double>> sampleStatistics)
    {
        DataStream<double[]> sampleScores =
            DataStreamUtils.mapPartition(
                sampleStatistics,
                new RichMapPartitionFunction<Tuple3<Double, Boolean, Double>, double[]>() {
                    @Override
                    public void mapPartition(
                        Iterable<Tuple3<Double, Boolean, Double>> dataPoints,
                        Collector<double[]> out) {
                        List<Tuple3<Double, Boolean, Double>> bufferedDataPoints =
                            new ArrayList<>();
                        for (Tuple3<Double, Boolean, Double> dataPoint : dataPoints) {
                            bufferedDataPoints.add(dataPoint);
                        }
                        double[] sampleScore = new double[NUM_SAMPLE];
                        Random rand = new Random();
                        for (int i = 0; i < NUM_SAMPLE; ++i) {
                            sampleScore[i] =
                                bufferedDataPoints.get(
                                    rand.nextInt(bufferedDataPoints.size()))
                                    .f0;
                        }
                        out.collect(sampleScore);
                    }
                });
        final int parallel = sampleScores.getParallelism();

        DataStream<double[]> rangeBoundary =
            DataStreamUtils.mapPartition(
                sampleScores,
                new RichMapPartitionFunction<double[], double[]>() {
                    @Override
                    public void mapPartition(
                        Iterable<double[]> dataPoints, Collector<double[]> out) {
                        double[] allSampleScore = new double[parallel * NUM_SAMPLE];
                        int cnt = 0;
                        for (double[] dataPoint : dataPoints) {
                            System.arraycopy(
                                dataPoint,
                                0,
                                allSampleScore,
                                cnt * NUM_SAMPLE,
                                NUM_SAMPLE);
                            cnt++;
                        }
                        Arrays.sort(allSampleScore);
                        double[] rangeBoundary = new double[parallel];
                        for (int i = 0; i < parallel; ++i) {
                            rangeBoundary[i] = allSampleScore[i * NUM_SAMPLE];
                        }
                        out.collect(rangeBoundary);
                    }
                });
        rangeBoundary.getTransformation().setParallelism(1);
        return rangeBoundary;
    }
}
