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

import org.apache.flink.ml.common.param.HasLabelCol;
import org.apache.flink.ml.common.param.HasSeed;
import org.apache.flink.ml.common.param.HasWeightCol;
import org.apache.flink.ml.param.BooleanParam;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Params for {@link RejectInference}.
 *
 * @param <T> The class type of this instance.
 */
public interface RejectInferenceParams<T> extends HasSeed<T>, HasWeightCol<T>, HasLabelCol<T> {
    Param<String> KNOWN_GOOD_BAD_SCORE_COL =
            new StringParam(
                    "knownGoodBadScoreCol",
                    "The prediction column of a Scorecard estimator. The prediction reflects the odds of "
                            + "an applicant being a good credit risk versus being a bad credit risk",
                    "knownGoodBadScoreCol");

    Param<String> ACCEPT_RATE_SCORE_COL =
            new StringParam(
                    "acceptRateScoreCol",
                    "The prediction column of an Linear estimator that estimates the probability of"
                            + " an applicant to be accepted",
                    "acceptRateScoreCol");

    Param<String> REJECTION_INFERENCE_METHOD =
            new StringParam(
                    "method",
                    "The method of rejection inference, there are fuzzy, hard_cutoff, parcelling and two_stage.",
                    RejectInferenceMethod.FUZZY.name(),
                    ParamValidators.inArray(
                            RejectInferenceMethod.FUZZY.name(),
                            RejectInferenceMethod.HARD_CUTOFF.name(),
                            RejectInferenceMethod.PARCELLING.name(),
                            RejectInferenceMethod.TWO_STAGE.name()));

    Param<Double> REJECTION_RATE =
            new DoubleParam(
                    "rejectionRate",
                    "The Rejection Rate represents the probability of rejection in the population.",
                    0.3,
                    ParamValidators.inRange(0, 1));

    Param<Double> EVENT_RATE_INCREASE =
            new DoubleParam(
                    "eventRateIncrease",
                    "The Event Rate Increase is a scaling entity that differs according to the selected Inference "
                            + "Method.",
                    1.0,
                    ParamValidators.inRange(0, 100));
    Param<Integer> NUM_BUCKETS =
            new IntParam("numBuckets", "Number of bucket to produce.", 25, ParamValidators.gtEq(2));

    Param<Integer> CUTOFF_SCORE =
            new IntParam(
                    "cutoffScore",
                    "Used in Hard Cutoff method to classify observations from the rejects data  as good or bad event.",
                    null);

    Param<String> SCORE_RANGE_METHOD =
            new StringParam(
                    "scoreRangeMethod",
                    "Use the Score Range Method property to specify the way that you want to define the range of "
                            + "scores to be bucketed.",
                    ScoreRangeMethod.ACCEPTS.name(),
                    ParamValidators.inArray(
                            ScoreRangeMethod.ACCEPTS.name(),
                            ScoreRangeMethod.REJECTS.name(),
                            ScoreRangeMethod.AUGMENTATION.name()));

    Param<Double> PDO =
            new DoubleParam(
                    "pdo",
                    "log(odds) = a * scaledValue + b, log(odds*2) = a * (scaledValue+pdo) + b",
                    null);

    Param<Double> ODDS =
            new DoubleParam(
                    "odds",
                    "log(odds) = a * scaledValue + b, log(odds*2) = a * (scaledValue+pdo) + b",
                    null);

    Param<Double> SCALED_VALUE =
            new DoubleParam(
                    "scaledValue",
                    "log(odds) = a * scaledValue + b, log(odds*2) = a * (scaledValue+pdo) + b",
                    null);

    Param<Boolean> WITH_SCALED =
            new BooleanParam(
                    "caseScaled",
                    "Whether the input scores and Cutoff Score value represent the log(odds), "
                            + "or the score have been transformed into a scaled score using the following formula:"
                            + " log(odds) = a * scaledScore + b",
                    false);

    default T setKnownGoodBadScoreCol(String value) {
        return set(KNOWN_GOOD_BAD_SCORE_COL, value);
    }

    default String getKnownGoodBadScoreCol() {
        return get(KNOWN_GOOD_BAD_SCORE_COL);
    }

    default T setAcceptRateScoreCol(String value) {
        return set(ACCEPT_RATE_SCORE_COL, value);
    }

    default String getAcceptRateScoreCol() {
        return get(ACCEPT_RATE_SCORE_COL);
    }

    default T setRejectionInferenceMethod(RejectInferenceMethod value) {
        return set(REJECTION_INFERENCE_METHOD, value.name());
    }

    default T setRejectionInferenceMethod(String value) {
        return set(REJECTION_INFERENCE_METHOD, value);
    }

    default RejectInferenceMethod getRejectionInferenceMethod() {
        return RejectInferenceMethod.valueOf(get(REJECTION_INFERENCE_METHOD));
    }

    default T setRejectionRate(double value) {
        return set(REJECTION_RATE, value);
    }

    default Double getRejectionRate() {
        return get(REJECTION_RATE);
    }

    default T setEventRateIncrease(double value) {
        return set(EVENT_RATE_INCREASE, value);
    }

    default Double getEventRateIncrease() {
        return get(EVENT_RATE_INCREASE);
    }

    default T setNumBuckets(int value) {
        return set(NUM_BUCKETS, value);
    }

    default int getNumBuckets() {
        return get(NUM_BUCKETS);
    }

    default T setCutoffScore(int value) {
        return set(CUTOFF_SCORE, value);
    }

    default Integer getCutoffScore() {
        return get(CUTOFF_SCORE);
    }

    default T setScoreRangeMethod(ScoreRangeMethod value) {
        return set(SCORE_RANGE_METHOD, value.name());
    }

    default T setScoreRangeMethod(String value) {
        return set(SCORE_RANGE_METHOD, value);
    }

    default ScoreRangeMethod getScoreRangeMethod() {
        return ScoreRangeMethod.valueOf(get(SCORE_RANGE_METHOD));
    }

    default T setPdo(double value) {
        return set(PDO, value);
    }

    default Double getPdo() {
        return get(PDO);
    }

    default T setOdds(double value) {
        return set(ODDS, value);
    }

    default Double getOdds() {
        return get(ODDS);
    }

    default T setScaledValue(double value) {
        return set(SCALED_VALUE, value);
    }

    default Double getScaledValue() {
        return get(SCALED_VALUE);
    }

    default T setWithScaled(boolean value) {
        return set(WITH_SCALED, value);
    }

    default boolean getWithScaled() {
        return get(WITH_SCALED);
    }

    /**
     * Reject Inference Method property is used to specify the method to classify rejects data set
     * observations.
     */
    enum RejectInferenceMethod {
        FUZZY,
        HARD_CUTOFF,
        PARCELLING,
        TWO_STAGE
    }

    /**
     * Score Range Method property is used to specify the way to define the range of scores to be
     * bucketed.
     */
    enum ScoreRangeMethod {
        ACCEPTS,
        REJECTS,
        AUGMENTATION
    }
}
