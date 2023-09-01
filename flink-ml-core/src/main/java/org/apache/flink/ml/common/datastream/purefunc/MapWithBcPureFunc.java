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

package org.apache.flink.ml.common.datastream.purefunc;

import org.apache.flink.annotation.Experimental;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.functions.RuntimeContext;

/**
 * Similar to {@link MapFunction} but with an addition broadcast parameter. Compared to {@link
 * RichMapFunction} with {@link RuntimeContext#getBroadcastVariable}, this interface can be used in
 * a broader situations since it involves no Flink runtime.
 *
 * @param <IN> Type of input elements.
 * @param <OUT> Type of output elements.
 * @param <BC> Type of broadcast element.
 */
@Experimental
@FunctionalInterface
public interface MapWithBcPureFunc<IN, OUT, BC> extends PureFunc {
    // TODO: use List<BC> instead of BC
    OUT map(IN elem, BC bc);
}
