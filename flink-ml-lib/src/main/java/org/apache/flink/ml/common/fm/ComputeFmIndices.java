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

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.ps.iterations.ProcessStage;
import org.apache.flink.ml.common.ps.sarray.SharedLongArray;

import it.unimi.dsi.fastutil.longs.LongOpenHashSet;

import java.util.Iterator;
import java.util.List;

/**
 * An iteration stage that samples a batch of training data and computes the indices needed to
 * compute gradients.
 */
public class ComputeFmIndices extends ProcessStage<FmMLSession> {

    private final int modelSize;

    public ComputeFmIndices(int modelSize) {
        this.modelSize = modelSize;
    }

    @Override
    public void process(FmMLSession session) throws Exception {

        session.readInNextBatchData();
        // Resets the offset of indices as zero.
        getUnSortedIndices(session.batchData, session.indices);
        // Resets the pulled value size as the same as indices.
        session.values.size(session.indices.size() * modelSize);
    }

    public static void getUnSortedIndices(List<FmSample> dataPoints, SharedLongArray indices) {
        LongOpenHashSet indexSet = new LongOpenHashSet();
        for (FmSample dataPoint : dataPoints) {
            Tuple2<long[], double[]> feature = dataPoint.features;
            long[] notZeros = feature.f0;
            for (long index : notZeros) {
                indexSet.add(index);
            }
        }
        indexSet.add(-1L);
        indices.size(indexSet.size());
        long[] elements = indices.elements();
        Iterator<Long> iterator = indexSet.iterator();
        int i = 0;
        while (iterator.hasNext()) {
            elements[i++] = iterator.next();
        }
    }
}
