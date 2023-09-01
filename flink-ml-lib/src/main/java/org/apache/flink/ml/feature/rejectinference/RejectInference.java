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

package org.apache.flink.ml.feature.rejectinference;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.apache.flink.table.api.Expressions.$;

/**
 * An implementation of reject reference to remedy the selection bias of credit scoring models. The
 * sample data that was used to develop a credit scoring model is structurally different from the
 * "through-the-door" population to which the credit scoring model is applied. The good/bad target
 * variable that is created for the credit scoring model is based on the records of applicants who
 * were all accepted for credit. However, the population to which the credit scoring model is
 * applied is composed includes applicants who would have been rejected under the scoring rules that
 * were used to generate the initial model.
 *
 * <p>There are four different methods that you can use to create inferred data from the rejects
 * data set: Fuzzy, Hard Cutoff, Parceling and Two-Stage. The following link is the introduction of
 * the algorithm by SAS software.
 *
 * <p><a href="https://documentation.sas.com/doc/en/emref/14.3/p07a3ma2a34qvqn1goqdq5y2dbfr.htm">SAS
 * helper</a>
 */
public class RejectInference
        implements AlgoOperator<RejectInference>, RejectInferenceParams<RejectInference> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private static final String INFERENCE_LABEL = "inferential_label";
    private static final String FREQUENCY_WEIGHT = "frequency_weight";
    private static final String ACCEPT_SUM = "ACCEPT_SUM";
    private static final String ACCEPT_NUM = "ACCEPT_NUM";
    private static final String REJECT_NUM = "REJECT_NUM";
    private static final String MIN_SCORE = "MIN_SCORE";
    private static final String MAX_SCORE = "MAC_SCORE";
    private static final String COVARIANCE = "COVARIANCE";
    private static final String K_BINS_DISCRETIZER_OUTPUT = "parcelling_bins_id";
    private static final String ODDS = "ODDS";
    private static final Logger LOG = LoggerFactory.getLogger(RejectInference.class);

    public RejectInference() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(2 == inputs.length);
        Table accepts = inputs[0];
        Table rejects = inputs[1];
        RowTypeInfo acceptsTypeInfo = TableUtils.getRowTypeInfo(accepts.getResolvedSchema());
        RowTypeInfo rejectsTypeInfo = TableUtils.getRowTypeInfo(rejects.getResolvedSchema());
        StreamTableEnvironment tenv =
                (StreamTableEnvironment) ((TableImpl) accepts).getTableEnvironment();
        DataStream<Row> acceptsStream = tenv.toDataStream(accepts);
        DataStream<Row> rejectsStream = tenv.toDataStream(rejects);

        final String knownGoodBadCol = getKnownGoodBadScoreCol();
        final String actualLabelCol = getLabelCol();
        final int rejectKgbScoreColIdx = rejectsTypeInfo.getFieldIndex(knownGoodBadCol);
        final RejectInferenceMethod method = getRejectionInferenceMethod();
        final double rejectionRate = getRejectionRate();
        final long seed = getSeed();
        final double eventIncreaseRate = getEventRateIncrease();
        final String weightCol = getWeightCol();
        double originCutoff = 0;
        if (RejectInferenceMethod.HARD_CUTOFF == method) {
            originCutoff = getCutoffScore();
        }

        // Convert scaled score to log(odds).
        if (getWithScaled()) {
            double odds = getOdds();
            double pdo = getPdo();
            double scaledValue = getScaledValue();
            final double scaledA = (Math.log(odds * 2) - Math.log(odds)) / pdo;
            final double scaledB = Math.log(odds) - scaledA * scaledValue;

            acceptsStream =
                    acceptsStream.map(
                            new ScaledValueConvertMapFunction(scaledA, scaledB, knownGoodBadCol),
                            acceptsTypeInfo);
            rejectsStream =
                    rejectsStream.map(
                            new ScaledValueConvertMapFunction(scaledA, scaledB, knownGoodBadCol),
                            rejectsTypeInfo);
            accepts = tenv.fromDataStream(acceptsStream);
            rejects = tenv.fromDataStream(rejectsStream);
            originCutoff = scaledA * originCutoff + scaledB;
        }
        final double cutoffScore = originCutoff;

        Map<String, DataStream<?>> broadcastMap = new HashMap<>(2);
        DataStream<Double> numAccept =
                DataStreamUtils.aggregate(acceptsStream, new SampleSumCountFunction(weightCol));
        DataStream<Double> numReject =
                DataStreamUtils.aggregate(rejectsStream, new SampleSumCountFunction(null));
        broadcastMap.put(ACCEPT_NUM, numAccept);
        broadcastMap.put(REJECT_NUM, numReject);
        numReject.print();
        numAccept.print();
        rejectsStream.print();

        if (RejectInferenceMethod.TWO_STAGE == method) {
            final String acceptRateCol = getAcceptRateScoreCol();
            // Model linear relationship between AR(AcceptRate) score and KGB(KnownGoodBad) score.
            Table arAndKgbScore;
            if (null == weightCol) {
                arAndKgbScore = accepts.select($(acceptRateCol), $(knownGoodBadCol));
            } else {
                arAndKgbScore =
                        accepts.select(
                                $(acceptRateCol),
                                $(knownGoodBadCol),
                                $(weightCol).cast(DataTypes.DOUBLE()).as(weightCol));
            }
            Map<String, DataStream<?>> statistics =
                    doSingleVariableLinearRegression(
                            arAndKgbScore,
                            tenv,
                            broadcastMap,
                            acceptRateCol,
                            knownGoodBadCol,
                            weightCol);

            // Convert AR score to KGB score.
            int arScoreColIdx = rejectsTypeInfo.getFieldIndex(acceptRateCol);
            rejectsStream =
                    BroadcastUtils.withBroadcastStream(
                            Collections.singletonList(rejectsStream),
                            statistics,
                            inputList -> {
                                DataStream input = inputList.get(0);
                                return input.map(
                                        new ArToKgbScoreMapFunction(
                                                rejectKgbScoreColIdx, arScoreColIdx),
                                        rejectsTypeInfo);
                            });
        }

        DataStream<Row> referenceStream;
        final RowTypeInfo referenceResultTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(rejectsTypeInfo.getFieldTypes(), Types.INT, Types.DOUBLE),
                        ArrayUtils.addAll(
                                rejectsTypeInfo.getFieldNames(),
                                INFERENCE_LABEL,
                                FREQUENCY_WEIGHT));

        // Parcelling method need to distribute the rejects into equal-sized buckets
        if (RejectInferenceMethod.PARCELLING == method
                || RejectInferenceMethod.TWO_STAGE == method) {
            Map<String, DataStream<?>> minMaxScore;
            final ScoreRangeMethod scoreRangeMethod = getScoreRangeMethod();

            DataStream<Row> acceptKgbStream = tenv.toDataStream(accepts.select($(knownGoodBadCol)));
            DataStream<Row> rejectKgbStream = tenv.toDataStream(rejects.select($(knownGoodBadCol)));
            if (ScoreRangeMethod.ACCEPTS == scoreRangeMethod) {
                minMaxScore = collectMinMaxScore(acceptKgbStream);
            } else if (ScoreRangeMethod.REJECTS == scoreRangeMethod) {
                minMaxScore = collectMinMaxScore(rejectKgbStream);
            } else {
                minMaxScore = collectMinMaxScore(acceptKgbStream, rejectKgbStream);
            }

            DataStream<Row> odds =
                    countLabelSumPerBucket(
                            acceptsStream,
                            acceptsTypeInfo,
                            minMaxScore,
                            getNumBuckets(),
                            knownGoodBadCol,
                            actualLabelCol);
            broadcastMap.put(ODDS, odds);
            DataStream<Row> discreteRejectStream =
                    transformDiscreteData(
                            rejectsStream,
                            rejectsTypeInfo,
                            minMaxScore,
                            getNumBuckets(),
                            knownGoodBadCol);
            referenceStream =
                    BroadcastUtils.withBroadcastStream(
                            Collections.singletonList(discreteRejectStream),
                            broadcastMap,
                            inputList -> {
                                DataStream input = inputList.get(0);
                                return input.flatMap(
                                        new ParcellingMethodMapFunction(
                                                seed,
                                                eventIncreaseRate,
                                                rejectionRate,
                                                getNumBuckets()),
                                        referenceResultTypeInfo);
                            });
        } else {
            referenceStream =
                    BroadcastUtils.withBroadcastStream(
                            Collections.singletonList(rejectsStream),
                            broadcastMap,
                            inputList -> {
                                DataStream input = inputList.get(0);
                                return input.flatMap(
                                        new FuzzyAndCutoffMethodMapFunction(
                                                method,
                                                cutoffScore,
                                                rejectionRate,
                                                rejectKgbScoreColIdx,
                                                eventIncreaseRate),
                                        referenceResultTypeInfo);
                            });
        }

        return new Table[] {tenv.fromDataStream(referenceStream)};
    }

    /**
     * Collect the minimal and maximal value of a certain column from multi DataStreams. These
     * DataStreams must have same types.
     *
     * @param scores The input DataStreams that have same types.
     * @return A map of DataStream that stores minimal and maximal value separately.
     */
    private Map<String, DataStream<?>> collectMinMaxScore(DataStream<Row>... scores) {
        DataStream<Row> maxScores = null;
        DataStream<Row> minScores = null;
        final int knownGoodBadCol = 0;
        for (DataStream<Row> score : scores) {
            DataStream maxScore =
                    DataStreamUtils.reduce(
                            score,
                            new ReduceFunction<Row>() {
                                @Override
                                public Row reduce(Row row, Row t1) throws Exception {
                                    double score1 = row.getFieldAs(knownGoodBadCol);
                                    double score2 = t1.getFieldAs(knownGoodBadCol);
                                    t1.setField(knownGoodBadCol, Math.max(score1, score2));
                                    return t1;
                                }
                            });
            DataStream minScore =
                    DataStreamUtils.reduce(
                            score,
                            (row, t1) -> {
                                double score1 = row.getFieldAs(knownGoodBadCol);
                                double score2 = t1.getFieldAs(knownGoodBadCol);
                                t1.setField(knownGoodBadCol, Math.min(score1, score2));
                                return t1;
                            });
            maxScores = null == maxScores ? maxScore : maxScores.union(maxScore);
            minScores = null == minScores ? minScore : minScores.union(minScore);
        }
        maxScores =
                DataStreamUtils.reduce(
                        maxScores,
                        (row, t1) -> {
                            t1.setField(
                                    knownGoodBadCol,
                                    Math.max(
                                            (double) row.getField(knownGoodBadCol),
                                            (double) t1.getField(knownGoodBadCol)));
                            return t1;
                        });
        minScores =
                DataStreamUtils.reduce(
                        minScores,
                        (row, t1) -> {
                            t1.setField(
                                    knownGoodBadCol,
                                    Math.min(
                                            (double) row.getField(knownGoodBadCol),
                                            (double) t1.getField(knownGoodBadCol)));
                            return t1;
                        });
        Map<String, DataStream<?>> broadcastNums = new HashMap<>(2);
        broadcastNums.put(MAX_SCORE, maxScores);
        broadcastNums.put(MIN_SCORE, minScores);
        return broadcastNums;
    }

    /*
     * Applies a {@link RichFlatMapFunction} on the reject data stream to infer
     * the posterior probabilities and the reject weights when using Parceling method.
     */
    private static class ParcellingMethodMapFunction extends RichFlatMapFunction<Row, Row> {
        private final double eventIncreaseRate, rejectionRate;
        private final Random random;
        private final int numBuckets;

        double[] oddsArray;
        double rejectWeights;

        public ParcellingMethodMapFunction(
                long seed, double eventIncreaseRate, double rejectionRate, int numBuckets) {
            this.random = new Random(seed);
            this.eventIncreaseRate = eventIncreaseRate;
            this.rejectionRate = rejectionRate;
            this.numBuckets = numBuckets;
        }

        /**
         * Calculating odds in each bucket.
         *
         * @param buckets A list of Row that consists of three parts: bucket number, number of
         *     positive samples, and number of negative samples. Some buckets may have no samples,
         *     which need to be filled with a value.
         * @return Odds od every bucket. The bucket has no samples is calculated by interpolating
         *     the results of the nearest two buckets.
         */
        private double[] calculateOddsInEachBucket(List<Row> buckets) {
            List<Integer> indices = new ArrayList<>();
            double[] odds = new double[numBuckets];
            for (Row r : buckets) {
                int bucketId = r.getFieldAs(0);
                int nGood = r.getFieldAs(1);
                int nBad = r.getFieldAs(2);
                if (bucketId >= 0) {
                    indices.add(bucketId);
                    odds[bucketId] = eventIncreaseRate * nBad / (nGood + nBad);
                }
            }
            if (indices.size() == numBuckets) {
                return odds;
            }
            if (indices.size() == 0) {
                throw new IllegalArgumentException(
                        "Input accepts data is empty, please make sure accepts has data.");
            }
            Integer[] indexArray = indices.toArray(new Integer[0]);
            Arrays.sort(indexArray);
            for (int i = 0; i < numBuckets; i++) {
                int pos = Arrays.binarySearch(indexArray, i);
                if (pos < 0) {
                    pos = -(pos + 1);
                    if (pos == 0) {
                        odds[i] = odds[indexArray[0]];
                    } else if (pos == indexArray.length) {
                        odds[i] = odds[indexArray[indexArray.length - 1]];
                    } else {
                        int preIndex = indexArray[pos - 1];
                        int nextIndex = indexArray[pos];
                        odds[i] =
                                (i - preIndex)
                                                * (odds[nextIndex] - odds[preIndex])
                                                / (nextIndex - preIndex)
                                        + odds[preIndex];
                    }
                }
                LOG.info("P(odds) of bucked {} is {}.", i, odds[i]);
            }
            return odds;
        }

        @Override
        public void flatMap(Row o, Collector collector) throws Exception {
            if (null == oddsArray) {
                oddsArray =
                        calculateOddsInEachBucket(getRuntimeContext().getBroadcastVariable(ODDS));
                double acceptNum =
                        ((Double) getRuntimeContext().getBroadcastVariable(ACCEPT_NUM).get(0));
                double rejectNum =
                        ((Double) getRuntimeContext().getBroadcastVariable(REJECT_NUM).get(0));
                rejectWeights = (rejectionRate / (1 - rejectionRate)) * (acceptNum / rejectNum);
                LOG.info(
                        "Sum weight of accepts data is {}. Sum weight of accepts data is {}.",
                        acceptNum,
                        rejectNum);
            }
            int bucketId = o.getFieldAs(K_BINS_DISCRETIZER_OUTPUT);
            if (bucketId < 0) {
                return;
            }
            if (random.nextDouble() < oddsArray[bucketId]) {
                o.setField(K_BINS_DISCRETIZER_OUTPUT, 0);
            } else {
                o.setField(K_BINS_DISCRETIZER_OUTPUT, 1);
            }
            collector.collect(Row.join(o, Row.of(rejectWeights)));
        }
    }

    /*
     * Applies a {@link RichFlatMapFunction} on the reject data stream to infer
     * the posterior probabilities and the reject weights when using Fuzzy or Hard-Cutoff method.
     */
    private static class FuzzyAndCutoffMethodMapFunction extends RichFlatMapFunction<Row, Row> {

        private final RejectInferenceMethod method;
        private final double cutoffScore;
        private final double rejectionRate;
        private final double eventIncRate;
        private final int rejectKgbScoreColIdx;
        private Double rejectWeight;

        public FuzzyAndCutoffMethodMapFunction(
                RejectInferenceMethod method,
                double cutoffScore,
                double rejectionRate,
                int rejectKgbScoreColIdx,
                double eventIncRate) {
            this.method = method;
            this.cutoffScore = cutoffScore;
            this.rejectionRate = rejectionRate;
            this.rejectKgbScoreColIdx = rejectKgbScoreColIdx;
            this.eventIncRate = eventIncRate;
        }

        @Override
        public void flatMap(Row row, Collector<Row> collector) throws Exception {
            if (null == rejectWeight) {
                double acceptNum =
                        (Double) getRuntimeContext().getBroadcastVariable(ACCEPT_NUM).get(0);
                double rejectNum =
                        (Double) getRuntimeContext().getBroadcastVariable(REJECT_NUM).get(0);
                rejectWeight = rejectionRate / (1 - rejectionRate) * acceptNum / rejectNum;
                LOG.info(
                        "Sum weight of accepts data is {}. Sum weight of accepts data is {}.",
                        acceptNum,
                        rejectNum);
                LOG.info("Reject weight is {}.", rejectWeight);
            }
            double score = row.getFieldAs(rejectKgbScoreColIdx);
            if (RejectInferenceMethod.HARD_CUTOFF == method) {
                int label = score < cutoffScore ? 0 : 1;
                collector.collect(Row.join(row, Row.of(label, rejectWeight)));
            } else {
                double frequencyWeightGood = rejectWeight * logOdds2Probability(score);
                collector.collect(Row.join(row, Row.of(1, frequencyWeightGood)));
                collector.collect(
                        Row.join(
                                row,
                                Row.of(0, (rejectWeight - frequencyWeightGood) * eventIncRate)));
            }
        }
    }

    /**
     * Do Linear Regression With One Variable.
     *
     * <p>Model the relationship between two variables by fitting an equation of the form Y = b +
     * wX. The formula for solving simple linear regression is as follows: w =
     * sum((x_i-x_bar)*(y_i-y_bar)) / sum(x_i-x_bar)^2 b = y_bar - w * x_bar
     *
     * @param table The training table.
     * @param tenv The StreamTableEnvironment of training table.
     * @param broadcastMap The map stores broadcast datastreams.
     * @param x The explanatory variable x represents P(accept).
     * @param y The dependent variable y represents logOdds.
     * @param weightCol The column of sample weight.
     * @return The slope and intercept of the linear regression line.
     */
    public static Map<String, DataStream<?>> doSingleVariableLinearRegression(
            Table table,
            StreamTableEnvironment tenv,
            Map<String, DataStream<?>> broadcastMap,
            final String x,
            final String y,
            @Nullable final String weightCol) {

        RowTypeInfo typeInfo = TableUtils.getRowTypeInfo(table.getResolvedSchema());
        DataStream<Row> stream = tenv.toDataStream(table);

        // Calculate accept probability. logOddsAccept = log(P(accept)/P(reject))
        stream =
                stream.map(
                        (row -> {
                            double logOddsAccept = row.getFieldAs(x);
                            row.setField(x, logOdds2Probability(logOddsAccept));
                            return row;
                        }),
                        typeInfo);

        DataStream<Row> sumStream =
                DataStreamUtils.reduce(
                        stream,
                        (o, t1) -> {
                            if (null != weightCol) {
                                double weight1 = ((Number) o.getFieldAs(weightCol)).doubleValue();
                                double weight2 = ((Number) t1.getFieldAs(weightCol)).doubleValue();
                                weight1 = weight1 < 0 ? 1 : weight1;
                                weight2 = weight2 < 0 ? 1 : weight2;
                                o.setField(
                                        x,
                                        weight1 * (double) o.getFieldAs(x)
                                                + weight2 * (double) t1.getFieldAs(x));
                                o.setField(
                                        y,
                                        weight1 * (double) o.getFieldAs(y)
                                                + weight2 * (double) t1.getFieldAs(y));
                                o.setField(weightCol, Double.NEGATIVE_INFINITY);
                            } else {
                                o.setField(x, (double) o.getFieldAs(x) + (double) t1.getFieldAs(x));
                                o.setField(y, (double) o.getFieldAs(y) + (double) t1.getFieldAs(y));
                            }
                            return o;
                        });
        broadcastMap.put(ACCEPT_SUM, sumStream);

        // Count how many rows that accept data set has.
        // DataStream<Row> numStream = DataStreamUtils.reduce(stream, new WeightedDataSetCounter(0,
        // false));
        DataStream<Row> intermediateResult =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(stream),
                        broadcastMap,
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return input.map(
                                    new RichMapFunction<Row, Row>() {
                                        private double[] avg;

                                        @Override
                                        public Row map(Row o) {
                                            if (null == avg) {
                                                double num =
                                                        ((Double)
                                                                getRuntimeContext()
                                                                        .getBroadcastVariable(
                                                                                ACCEPT_NUM)
                                                                        .get(0));
                                                Row sum =
                                                        ((Row)
                                                                getRuntimeContext()
                                                                        .getBroadcastVariable(
                                                                                ACCEPT_SUM)
                                                                        .get(0));
                                                avg = new double[2];
                                                avg[0] = (double) sum.getFieldAs(x) / num;
                                                avg[1] = (double) sum.getFieldAs(y) / num;
                                            }
                                            // Calculate x-x_bar, y-y_bar
                                            double valueX = o.getFieldAs(x);
                                            double valueY = o.getFieldAs(y);
                                            double x = valueX - avg[0];
                                            double y = valueY - avg[1];
                                            if (null != weightCol) {
                                                double weight =
                                                        ((Number) o.getFieldAs(weightCol))
                                                                .doubleValue();
                                                o.setField(0, valueX * y * weight * weight);
                                                o.setField(1, valueX * x * weight * weight);
                                            } else {
                                                o.setField(0, valueX * y);
                                                o.setField(1, valueX * x);
                                            }
                                            return o;
                                        }
                                    },
                                    typeInfo);
                        });

        // Calculate  sum((x_i-x_bar)*(y_i-y_bar)) and sum(x_i-x_bar)^2.
        DataStream<Row> covariance =
                DataStreamUtils.reduce(
                        intermediateResult,
                        (o, t1) -> {
                            o.setField(0, (double) o.getFieldAs(0) + (double) t1.getFieldAs(0));
                            o.setField(1, (double) o.getFieldAs(1) + (double) t1.getFieldAs(1));
                            return o;
                        });
        broadcastMap.put(COVARIANCE, covariance);
        return broadcastMap;
    }

    /**
     * Assign credible performance to each unknown rejected samples.
     *
     * <p>The formula is as follows: P(good) = P(good|accept)*P(accept)+P(good|reject)P(reject)
     * P(good|accept) = KGB_SCORE P(accept) = AR_SCORE P(good|reject) = b + w * P(accept) The
     * coefficient w and b are results of `singleVariableLinearRegression`.
     */
    public static class ArToKgbScoreMapFunction extends RichMapFunction<Row, Row> {
        private int kgbScoreColIdx;
        private int arScoreColIdx;
        private double[] avg;
        private double w;
        private double b;

        public ArToKgbScoreMapFunction(int kgbScoreColIdx, int arScoreColIdx) {
            this.kgbScoreColIdx = kgbScoreColIdx;
            this.arScoreColIdx = arScoreColIdx;
        }

        @Override
        public Row map(Row row) {
            if (null == avg) {
                double num = ((Double) getRuntimeContext().getBroadcastVariable(ACCEPT_NUM).get(0));
                Row sum = ((Row) getRuntimeContext().getBroadcastVariable(ACCEPT_SUM).get(0));
                avg = new double[2];
                avg[0] = (double) sum.getFieldAs(0) / num;
                avg[1] = (double) sum.getFieldAs(1) / num;
                Row cov = ((Row) getRuntimeContext().getBroadcastVariable(COVARIANCE).get(0));
                w = (double) cov.getFieldAs(0) / (double) cov.getFieldAs(1);
                b = avg[1] - w * avg[0];
                LOG.info("The value of slope is {}. The value of intercept is {}.", w, b);
            }

            // P(good) = P(good|accept)*P(accept)+P(good|reject)P(reject)
            double goodAccept = logOdds2Probability(row.getFieldAs(kgbScoreColIdx));
            double accept = logOdds2Probability(row.getFieldAs(arScoreColIdx));
            double goodReject = logOdds2Probability(w * accept + b);
            double score = goodAccept * accept + goodReject * (1 - accept);
            row.setField(kgbScoreColIdx, probability2LogOdds(score));

            return row;
        }
    }

    /**
     * Transform continuous score into discrete ones.
     *
     * @param stream The input stream.
     * @param typeInfo The types of the stream.
     * @param minMaxScore The range of buckets.
     * @param numBucket The number of buckets.
     * @param knownGoodBadCol The name of score column.
     * @return
     */
    private DataStream<Row> transformDiscreteData(
            DataStream stream,
            RowTypeInfo typeInfo,
            Map<String, DataStream<?>> minMaxScore,
            final int numBucket,
            final String knownGoodBadCol) {
        RowTypeInfo withBucketIdTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(typeInfo.getFieldTypes(), Types.INT),
                        ArrayUtils.addAll(typeInfo.getFieldNames(), K_BINS_DISCRETIZER_OUTPUT));
        return BroadcastUtils.withBroadcastStream(
                Collections.singletonList(stream),
                minMaxScore,
                inputList -> {
                    DataStream input = inputList.get(0);
                    return input.map(
                            new RichMapFunction<Row, Row>() {
                                double[] minMaxScore;
                                double bucketWidth;

                                @Override
                                public Row map(Row o) throws Exception {
                                    if (null == minMaxScore) {
                                        minMaxScore = new double[2];
                                        minMaxScore[0] =
                                                ((Row)
                                                                getRuntimeContext()
                                                                        .getBroadcastVariable(
                                                                                MIN_SCORE)
                                                                        .get(0))
                                                        .getFieldAs(knownGoodBadCol);
                                        minMaxScore[1] =
                                                ((Row)
                                                                getRuntimeContext()
                                                                        .getBroadcastVariable(
                                                                                MAX_SCORE)
                                                                        .get(0))
                                                        .getFieldAs(knownGoodBadCol);
                                        bucketWidth = (minMaxScore[1] - minMaxScore[0]) / numBucket;
                                        LOG.info(
                                                "Range of bucket is [{},{}], num of buckets is {}.",
                                                minMaxScore[0],
                                                minMaxScore[1],
                                                numBucket);
                                    }
                                    double score = o.getFieldAs(knownGoodBadCol);
                                    if (score < minMaxScore[0] || score > minMaxScore[1]) {
                                        return Row.join(o, Row.of(-1));
                                    } else {
                                        int bucketId =
                                                (int)
                                                        Math.min(
                                                                numBucket - 1,
                                                                Math.floor(
                                                                        (score - minMaxScore[0])
                                                                                / bucketWidth));
                                        return Row.join(o, Row.of(bucketId));
                                    }
                                }
                            },
                            withBucketIdTypeInfo);
                });
    }

    /** Count sum of good and bad accepts in each bucket. */
    private DataStream<Row> countLabelSumPerBucket(
            DataStream stream,
            RowTypeInfo acceptsTypeInfo,
            Map<String, DataStream<?>> minMaxScore,
            final int numBucket,
            final String knownGoodBadCol,
            final String actualLabelCol) {
        DataStream<Row> discreteStream =
                transformDiscreteData(
                        stream, acceptsTypeInfo, minMaxScore, numBucket, knownGoodBadCol);
        int bucketIdPos = discreteStream.getType().getArity() - 1;
        return DataStreamUtils.keyedAggregate(
                discreteStream.keyBy(row -> (int) row.getField(bucketIdPos)),
                new AggregateFunction<Row, Tuple3<Integer, Integer, Integer>, Row>() {

                    @Override
                    public Tuple3<Integer, Integer, Integer> createAccumulator() {

                        return Tuple3.of(0, 0, 0);
                    }

                    @Override
                    public Tuple3<Integer, Integer, Integer> add(
                            Row row, Tuple3<Integer, Integer, Integer> acc) {

                        acc.f0 = row.getFieldAs(K_BINS_DISCRETIZER_OUTPUT);
                        if (1 == ((Number) row.getFieldAs(actualLabelCol)).intValue()) {
                            acc.f1++;
                        } else {
                            acc.f2++;
                        }
                        return acc;
                    }

                    @Override
                    public Row getResult(Tuple3<Integer, Integer, Integer> acc) {
                        return Row.of(acc.f0, acc.f1, acc.f2);
                    }

                    @Override
                    public Tuple3<Integer, Integer, Integer> merge(
                            Tuple3<Integer, Integer, Integer> t3,
                            Tuple3<Integer, Integer, Integer> acc1) {
                        acc1.f1 += t3.f1;
                        acc1.f2 += t3.f2;
                        return acc1;
                    }
                },
                Types.TUPLE(Types.INT, Types.INT, Types.INT),
                Types.ROW(Types.INT, Types.INT, Types.INT));
    }

    private static double logOdds2Probability(double logOdds) {
        return 1.0 / (1 + Math.exp(-logOdds));
    }

    private static double probability2LogOdds(double probability) {
        return Math.log(probability / (1 - probability));
    }

    private static class ScaledValueConvertMapFunction implements MapFunction<Row, Row> {

        private final double a;
        private final double b;
        private final String field;

        public ScaledValueConvertMapFunction(double a, double b, String field) {
            this.a = a;
            this.b = b;
            this.field = field;
        }

        @Override
        public Row map(Row row) throws Exception {
            double score = row.getFieldAs(field);
            row.setField(field, score * a + b);
            return row;
        }
    }

    private static class SampleSumCountFunction implements AggregateFunction<Row, Double, Double> {

        private final String weightCol;

        private SampleSumCountFunction(@Nullable String weightCol) {
            this.weightCol = weightCol;
        }

        @Override
        public Double createAccumulator() {
            return 0.;
        }

        @Override
        public Double add(Row value, Double accumulator) {
            double weight =
                    null != weightCol ? ((Number) value.getFieldAs(weightCol)).doubleValue() : 1;
            return accumulator + weight;
        }

        @Override
        public Double getResult(Double accumulator) {
            return accumulator;
        }

        @Override
        public Double merge(Double a, Double b) {
            return a + b;
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
