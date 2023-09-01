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

package org.apache.flink.ml.feature.transform;

import org.apache.flink.ml.param.BooleanParam;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.FloatParam;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.LongParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringArrayParam;
import org.apache.flink.ml.param.StringParam;
import org.apache.flink.ml.param.WithParams;

/**
 * Params of {@link TypeTransform}.
 *
 * @param <T> The class type of this instance.
 */
public interface TypeTransformParams<T> extends WithParams<T> {

    Param<Double> DEFAULT_DOUBLE_VALUE =
            new DoubleParam(
                    "defaultDoubleValue",
                    "The default double value.",
                    0.0,
                    ParamValidators.alwaysTrue());

    Param<Float> DEFAULT_FLOAT_VALUE =
            new FloatParam(
                    "defaultFloatValue",
                    "The default float value.",
                    0.0F,
                    ParamValidators.alwaysTrue());
    Param<Integer> DEFAULT_INT_VALUE =
            new IntParam(
                    "defaultIntValue", "The default int value.", 0, ParamValidators.alwaysTrue());

    Param<Long> DEFAULT_LONG_VALUE =
            new LongParam(
                    "defaultLongValue",
                    "The default long value.",
                    0L,
                    ParamValidators.alwaysTrue());

    Param<String> DEFAULT_STRING_VALUE =
            new StringParam(
                    "defaultStringValue",
                    "The default string value.",
                    "",
                    ParamValidators.alwaysTrue());

    Param<String[]> TO_DOUBLE_COLS =
            new StringArrayParam(
                    "toDoubleCols",
                    "Input column names to double.",
                    new String[] {},
                    ParamValidators.alwaysTrue());
    Param<String[]> TO_FLOAT_COLS =
            new StringArrayParam(
                    "toFloatCols",
                    "Input column names to float.",
                    new String[] {},
                    ParamValidators.alwaysTrue());
    Param<String[]> TO_INT_COLS =
            new StringArrayParam(
                    "toIntCols",
                    "Input column names to int.",
                    new String[] {},
                    ParamValidators.alwaysTrue());
    Param<String[]> TO_LONG_COLS =
            new StringArrayParam(
                    "toLongCols",
                    "Input column names to long.",
                    new String[] {},
                    ParamValidators.alwaysTrue());
    Param<String[]> TO_STRING_COLS =
            new StringArrayParam(
                    "toStringCols",
                    "Input column names to string.",
                    new String[] {},
                    ParamValidators.alwaysTrue());

    Param<Boolean> KEEP_OLD_COLS =
            new BooleanParam("keepOldCols", "Whether to keep the old columns.", false);

    default Double getDefaultDoubleValue() {
        return get(DEFAULT_DOUBLE_VALUE);
    }

    default T setDefaultDoubleValue(Double value) {
        return set(DEFAULT_DOUBLE_VALUE, value);
    }

    default Float getDefaultFloatValue() {
        return get(DEFAULT_FLOAT_VALUE);
    }

    default T setDefaultFloatValue(Float value) {
        return set(DEFAULT_FLOAT_VALUE, value);
    }

    default Integer getDefaultIntValue() {
        return get(DEFAULT_INT_VALUE);
    }

    default T setDefaultIntValue(Integer value) {
        return set(DEFAULT_INT_VALUE, value);
    }

    default Long getDefaultLongValue() {
        return get(DEFAULT_LONG_VALUE);
    }

    default T setDefaultLongValue(Long value) {
        return set(DEFAULT_LONG_VALUE, value);
    }

    default String getDefaultStringValue() {
        return get(DEFAULT_STRING_VALUE);
    }

    default T setDefaultStringValue(String value) {
        return set(DEFAULT_STRING_VALUE, value);
    }

    default String[] getToDoubleCols() {
        return get(TO_DOUBLE_COLS);
    }

    default T setToDoubleCols(String... value) {
        return set(TO_DOUBLE_COLS, value);
    }

    default String[] getToFloatCols() {
        return get(TO_FLOAT_COLS);
    }

    default T setToFloatCols(String... value) {
        return set(TO_FLOAT_COLS, value);
    }

    default String[] getToIntCols() {
        return get(TO_INT_COLS);
    }

    default T setToIntCols(String... value) {
        return set(TO_INT_COLS, value);
    }

    default String[] getToLongCols() {
        return get(TO_LONG_COLS);
    }

    default T setToLongCols(String... value) {
        return set(TO_LONG_COLS, value);
    }

    default String[] getToStringCols() {
        return get(TO_STRING_COLS);
    }

    default T setToStringCols(String... value) {
        return set(TO_STRING_COLS, value);
    }

    default boolean getKeepOldCols() {
        return get(KEEP_OLD_COLS);
    }

    default T setKeepOldCols(boolean value) {
        return set(KEEP_OLD_COLS, value);
    }
}
