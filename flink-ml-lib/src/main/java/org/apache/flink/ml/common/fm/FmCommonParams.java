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

package org.apache.flink.ml.common.fm;

import org.apache.flink.ml.common.param.HasGlobalBatchSize;
import org.apache.flink.ml.common.param.HasLabelCol;
import org.apache.flink.ml.common.param.HasMaxIter;
import org.apache.flink.ml.common.param.HasTol;
import org.apache.flink.ml.common.param.HasWeightCol;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/** Params for fm training. */
public interface FmCommonParams<T>
        extends HasLabelCol<T>, HasWeightCol<T>, HasGlobalBatchSize<T>, HasMaxIter<T>, HasTol<T> {

    Param<String> DIM = new StringParam("dim", "dim", "1,1,10", ParamValidators.notNull());

    Param<String> LAMBDA =
            new StringParam("lambda", "lambda", "0.01,0.01,0.01", ParamValidators.notNull());

    Param<Double> INIT_STD_EV =
            new DoubleParam("initStdEv", "init stdEv", 0.05, ParamValidators.gt(0.0));

    Param<Double> LEARN_RATE =
            new DoubleParam("learnRate", "learn rate", 0.01, ParamValidators.gt(0.0));

    Param<String> METHOD =
            new StringParam("method", "optimize method", "AdaGrad", ParamValidators.notNull());

    Param<Double> ALPHA =
            new DoubleParam(
                    "alpha",
                    "The alpha parameter of FTRL optimizer.",
                    0.1,
                    ParamValidators.gt(0.0));

    Param<Double> BETA =
            new DoubleParam(
                    "beta", "The beta parameter of FTRL optimizer.", 0.1, ParamValidators.gt(0.0));

    Param<Double> L_1 =
            new DoubleParam(
                    "l1", "The l1 parameter of FTRL optimizer.", 0.1, ParamValidators.gt(0.0));

    Param<Double> L_2 =
            new DoubleParam(
                    "l2", "The l2 parameter of FTRL optimizer.", 0.1, ParamValidators.gt(0.0));

    Param<Double> GAMMA =
            new DoubleParam(
                    "gamma",
                    "The gamma parameter of RMSProp or AdaDelta optimizer.",
                    0.9,
                    ParamValidators.gt(0.0));

    Param<Double> BETA_1 =
            new DoubleParam(
                    "beta1",
                    "The beta1 parameter of adam optimizer.",
                    0.9,
                    ParamValidators.gt(0.0));

    Param<Double> BETA_2 =
            new DoubleParam(
                    "beta2",
                    "The beta2 parameter of FTRL optimizer.",
                    0.999,
                    ParamValidators.gt(0.0));

    default String getDim() {
        return get(DIM);
    }

    default T setDim(String value) {
        return set(DIM, value);
    }

    default String getLambda() {
        return get(LAMBDA);
    }

    default T setLambda(String value) {
        return set(LAMBDA, value);
    }

    default Double getInitStdEv() {
        return get(INIT_STD_EV);
    }

    default T setInitStdEv(Double value) {
        return set(INIT_STD_EV, value);
    }

    default Double getLearnRate() {
        return get(LEARN_RATE);
    }

    default T setLearnRate(Double value) {
        return set(LEARN_RATE, value);
    }

    default String getMethod() {
        return get(METHOD);
    }

    default T setMethod(String value) {
        return set(METHOD, value);
    }

    default double getAlpha() {
        return get(ALPHA);
    }

    default T setAlpha(Double value) {
        return set(ALPHA, value);
    }

    default double getBeta() {
        return get(BETA);
    }

    default T setBeta(Double value) {
        return set(BETA, value);
    }

    default double getL1() {
        return get(L_1);
    }

    default T setL1(Double value) {
        return set(L_1, value);
    }

    default double getL2() {
        return get(L_2);
    }

    default T setL2(Double value) {
        return set(L_2, value);
    }

    default double getGamma() {
        return get(GAMMA);
    }

    default T setGamma(Double value) {
        return set(GAMMA, value);
    }

    default double getBeta1() {
        return get(BETA_1);
    }

    default T setBeta1(Double value) {
        return set(BETA_1, value);
    }

    default double getBeta2() {
        return get(BETA_2);
    }

    default T setBeta2(Double value) {
        return set(BETA_2, value);
    }
}
