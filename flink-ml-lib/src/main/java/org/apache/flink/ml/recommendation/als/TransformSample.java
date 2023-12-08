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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.types.Row;

/** Comments. */
public class TransformSample implements MapFunction<Row, Tuple3<Long, Long, Double>> {
    final String userCol;
    final String itemCol;
    final String ratingCol;

    public TransformSample(String userCol, String itemCol, String ratingCol) {
        this.userCol = userCol;
        this.itemCol = itemCol;
        this.ratingCol = ratingCol;
    }

    @Override
    public Tuple3<Long, Long, Double> map(Row value) throws Exception {
        Long user = value.getFieldAs(userCol);
        Long item = value.getFieldAs(itemCol);
        user = 2L * user;
        item = 2L * item + 1L;
        Number rating = ratingCol == null ? 0.0F : value.getFieldAs(ratingCol);

        return new Tuple3<>(user, item, rating.doubleValue());
    }
}
