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

package org.apache.flink.ml.recommendation.als;

import org.apache.flink.ml.common.param.HasSeed;
import org.apache.flink.ml.param.BooleanParam;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Params for {@link AlsModel}.
 *
 * @param <T> The class type of this instance.
 */
public interface AlsParams<T> extends HasSeed<T>, AlsModelParams<T> {

    Param<String> RATING_COL =
            new StringParam(
                    "ratingCol", "Column name for rating.", null, ParamValidators.notNull());

    Param<Double> ALPHA =
            new DoubleParam(
                    "alpha", "Alpha for implicit preference.", 1.0, ParamValidators.gtEq(0));

    Param<Double> REG_PARAM =
            new DoubleParam("regParam", "Regularization parameter.", 0.1, ParamValidators.gtEq(0.));

    Param<Boolean> IMPLICITPREFS =
            new BooleanParam(
                    "implicitPrefs",
                    "Whether to use implicit preference.",
                    false,
                    ParamValidators.alwaysTrue());

    Param<Boolean> NONNEGATIVE =
            new BooleanParam(
                    "nonnegative",
                    "Whether to use nonnegative constraint for least squares.",
                    false,
                    ParamValidators.alwaysTrue());

    Param<Integer> RANK =
            new IntParam("rank", "Rank of the factorization.", 10, ParamValidators.gt(0));

    Param<Integer> MAX_ITER =
            new IntParam("maxIter", "Maximum number of iterations.", 10, ParamValidators.gt(0));

    Param<Integer> NUM_USER_BLOCKS =
            new IntParam("numUserBlocks", "Number of user blocks.", 10, ParamValidators.gt(0));

    Param<Integer> NUM_ITEM_BLOCKS =
            new IntParam("numItemBlocks", "Number of item blocks.", 10, ParamValidators.gt(0));

    default String getRatingCol() {
        return get(RATING_COL);
    }

    default T setRatingCol(String value) {
        return set(RATING_COL, value);
    }

    default double getAlpha() {
        return get(ALPHA);
    }

    default T setAlpha(Double value) {
        return set(ALPHA, value);
    }

    default double getRegParam() {
        return get(REG_PARAM);
    }

    default T setRegParam(Double value) {
        return set(REG_PARAM, value);
    }

    default Boolean getImplicitprefs() {
        return get(IMPLICITPREFS);
    }

    default T setImplicitprefs(Boolean value) {
        return set(IMPLICITPREFS, value);
    }

    default Boolean getNonnegative() {
        return get(NONNEGATIVE);
    }

    default T setNonnegative(Boolean value) {
        return set(NONNEGATIVE, value);
    }

    default int getRank() {
        return get(RANK);
    }

    default T setRank(int value) {
        return set(RANK, value);
    }

    default int getMaxIter() {
        return get(MAX_ITER);
    }

    default T setMaxIter(int value) {
        return set(MAX_ITER, value);
    }

    default int getNumUserBlocks() {
        return get(NUM_USER_BLOCKS);
    }

    default T setNumUserBlocks(int value) {
        return set(NUM_USER_BLOCKS, value);
    }

    default int getNumItemBlocks() {
        return get(NUM_ITEM_BLOCKS);
    }

    default T setNumItemBlocks(int value) {
        return set(NUM_ITEM_BLOCKS, value);
    }
}
