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

package org.apache.flink.ml.recommendation.alsold;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.common.ps.api.function.GroupReduceFunction;
import org.apache.flink.ml.recommendation.alsold.Als.Ratings;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.List;

/** Comments. */
public class GenerateRatings
        extends GroupReduceFunction<Tuple3<Long, Long, Double>, Ratings, Long> {

    private final int threshold;

    public GenerateRatings(int threshold) {
        this.threshold = threshold;
    }

    @Override
    public void reduce(
            Iterable<Tuple3<Long, Long, Double>> iterable, Collector<Ratings> collector) {
        long srcNodeId = -1L;
        List<Tuple2<Long, Double>> neighbors = new ArrayList<>();

        for (Tuple3<Long, Long, Double> t4 : iterable) {
            srcNodeId = t4.f0;
            neighbors.add(Tuple2.of(t4.f1, t4.f2));
        }
        if (neighbors.size() > threshold) {
            int numBlock =
                    neighbors.size() / threshold + (neighbors.size() % threshold == 0L ? 0 : 1);
            int blockSize = neighbors.size() / numBlock;
            int startIndex = 0;
            for (int i = 0; i < numBlock; ++i) {
                Ratings tmpRating = new Ratings();
                int offset = Math.min(i + 1, neighbors.size() % numBlock);
                int endIndex = Math.min(neighbors.size(), (i + 1) * blockSize + offset);
                int size = endIndex - startIndex;
                tmpRating.neighbors = new long[size];
                tmpRating.scores = new double[size];
                for (int j = 0; j < size; j++) {
                    tmpRating.neighbors[j] = neighbors.get(startIndex + j).f0;
                    tmpRating.scores[j] = neighbors.get(startIndex + j).f1;
                }
                startIndex = endIndex;
                tmpRating.nodeId = srcNodeId;
                tmpRating.isMainNode = (i == 0);
                tmpRating.isSplit = true;
                tmpRating.numNeighbors = neighbors.size();
                collector.collect(tmpRating);
            }
        } else {
            Ratings returnRatings = new Ratings();
            returnRatings.nodeId = srcNodeId;
            returnRatings.neighbors = new long[neighbors.size()];
            returnRatings.scores = new double[neighbors.size()];
            returnRatings.isSplit = false;
            returnRatings.numNeighbors = neighbors.size();
            returnRatings.isMainNode = false;
            for (int i = 0; i < returnRatings.neighbors.length; i++) {
                returnRatings.neighbors[i] = neighbors.get(i).f0;
                returnRatings.scores[i] = neighbors.get(i).f1;
            }
            collector.collect(returnRatings);
        }
    }
}
