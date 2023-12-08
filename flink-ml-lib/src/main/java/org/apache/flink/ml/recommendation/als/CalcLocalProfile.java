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

import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.java.tuple.Tuple5;
import org.apache.flink.ml.recommendation.als.Als.Ratings;
import org.apache.flink.util.Collector;

/** Comments. */
public class CalcLocalProfile
        implements MapPartitionFunction<Ratings, Tuple5<Long, Long, Long, Integer, Integer>> {

    @Override
    public void mapPartition(
            Iterable<Ratings> values,
            Collector<Tuple5<Long, Long, Long, Integer, Integer>> collector) {
        long numUsers = 0L;
        long numItems = 0L;
        long numRatings = 0L;
        int hottestUserPoint = 0;
        int hottestItemPoint = 0;
        for (Ratings ratings : values) {
            if (ratings.nodeId % 2L == 0L) {
                numUsers++;
                numRatings += ratings.scores.length;
                hottestUserPoint = Math.max(hottestUserPoint, ratings.numNeighbors);
            } else {
                numItems++;
                hottestItemPoint = Math.max(hottestItemPoint, ratings.numNeighbors);
            }
        }
        collector.collect(
                Tuple5.of(numUsers, numItems, numRatings, hottestUserPoint, hottestItemPoint));
    }
}
