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

package org.apache.flink.ml.common.param;

import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringArrayParam;
import org.apache.flink.ml.param.WithParams;

/**
 * Interface for the shared featuresCols param.
 *
 * <p>{@link HasFeaturesCols} is typically used for stages that implement {@link HasLabelCol}. It is
 * preferred to use {@link HasInputCol} for other cases.
 */
public interface HasFeaturesCols<T> extends WithParams<T> {
    Param<String[]> FEATURES_COLS =
            new StringArrayParam(
                    "featuresCols",
                    "Feature column names.",
                    new String[] {},
                    ParamValidators.notNull());

    default String[] getFeaturesCols() {
        return get(FEATURES_COLS);
    }

    default T setFeaturesCols(String... value) {
        return set(FEATURES_COLS, value);
    }
}
