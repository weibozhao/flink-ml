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

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.util.Collector;

/** Comments. */
public class DuplicateSample
        implements FlatMapFunction<Tuple3<Long, Long, Double>, Tuple3<Long, Long, Double>> {

    @Override
    public void flatMap(
            Tuple3<Long, Long, Double> value, Collector<Tuple3<Long, Long, Double>> out) {
        out.collect(Tuple3.of(value.f0, value.f1, value.f2));
        out.collect(Tuple3.of(value.f1, value.f0, value.f2));
    }
}
