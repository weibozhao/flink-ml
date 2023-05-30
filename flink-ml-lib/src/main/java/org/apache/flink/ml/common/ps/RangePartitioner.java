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

package org.apache.flink.ml.common.ps;

import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.util.Preconditions;

import javax.annotation.Nullable;

import java.util.Arrays;
import java.util.Iterator;

/**
 * Range partitioner for model data. It partitions the model data for each dimension according to
 * the dimension id. The model data for each dimension could be a double or several doubles. Note
 * that the model data for all dimensions should share the same size.
 */
public class RangePartitioner {
    public final long dim;
    public final int numServers;
    public final long[] ranges;

    public RangePartitioner(long dim, int numServers) {
        Preconditions.checkArgument(
                dim > 0,
                String.format(
                        "Illegal dimension when using %s: %d",
                        RangePartitioner.class.getSimpleName(), dim));

        this.dim = dim;
        this.numServers = numServers;
        this.ranges = new long[numServers + 1];
        long shardSize = dim / numServers;

        for (int serverId = 0; serverId < numServers; serverId++) {
            ranges[serverId] = shardSize * serverId;
        }
        ranges[numServers] = dim;
    }

    /**
     * Splits the push/pull request according to the given sorted indices and the corresponding
     * values.
     *
     * @param indices sorted indices of push/pull request.
     * @param values the push values if not null.
     * @return the split requests for each server.
     */
    public Iterator<Tuple3<Integer, long[], double[]>> splitRequest(
            long[] indices, @Nullable double[] values) {
        return new RequestsIterator(numServers, indices, values, ranges);
    }

    private static class RequestsIterator implements Iterator<Tuple3<Integer, long[], double[]>> {
        private final int numServers;
        private final long[] indices;
        private final double[] values;
        /**
         * Number of values per key. If the model data is a vector, numValuesPerKey is one. If the
         * model data is a matrix, numValuesPerKey is the number of columns.
         */
        private final int numValuesPerKey;

        private final long[] ranges;

        private int serverId = 0;

        private int s = 0;

        public RequestsIterator(
                int numPss, long[] indices, @Nullable double[] values, long[] ranges) {
            this.numServers = numPss;
            this.indices = indices;
            this.values = values;
            this.ranges = ranges;
            if (indices.length != 0 && values != null) {
                numValuesPerKey = values.length / indices.length;
                Preconditions.checkArgument(
                        numValuesPerKey * indices.length == values.length,
                        String.format(
                                "The size of values [%d] cannot be divided by size of keys [%d].",
                                values.length, indices.length));
            } else {
                numValuesPerKey = 1;
            }
        }

        @Override
        public boolean hasNext() {
            return serverId < numServers;
        }

        @Override
        public Tuple3<Integer, long[], double[]> next() {
            int e = s;
            while (e < indices.length && indices[e] < ranges[serverId + 1]) {
                e++;
            }

            long[] splitIndices = new long[0];
            double[] splitValues = values == null ? null : new double[0];
            if (s < e) {
                splitIndices = Arrays.copyOfRange(indices, s, e);
                splitValues =
                        values == null
                                ? null
                                : Arrays.copyOfRange(
                                        values, s * numValuesPerKey, e * numValuesPerKey);
            }
            s = e;
            serverId++;
            return Tuple3.of(serverId - 1, splitIndices, splitValues);
        }
    }
}
