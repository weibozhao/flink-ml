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

package org.apache.flink.ml.common.gbt.operators;

import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple5;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.util.Collector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;

/**
 * This operator reduces histograms for (nodeId, featureId) pairs.
 *
 * <p>The input elements are tuples of (subtask index, (nodeId, featureId) pair index, sliceId,
 * numSlices, Histogram). The output elements are tuples of ((nodeId, featureId) pair index,
 * Histogram).
 */
public class ReduceHistogramFunction
        extends RichFlatMapFunction<
                Tuple5<Integer, Integer, Integer, Integer, Histogram>, Tuple2<Integer, Histogram>> {

    private static final Logger LOG = LoggerFactory.getLogger(ReduceHistogramFunction.class);
    private final Map<Integer, AccumulateState> pairState = new HashMap<>();
    private int numSubtasks;

    @Override
    public void open(Configuration parameters) throws Exception {
        numSubtasks = getRuntimeContext().getNumberOfParallelSubtasks();
    }

    @Override
    public void flatMap(
            Tuple5<Integer, Integer, Integer, Integer, Histogram> value,
            Collector<Tuple2<Integer, Histogram>> out)
            throws Exception {
        int sourceSubtaskId = value.f0;
        int pairId = value.f1;
        int numSlices = value.f3;
        Histogram histogram = value.f4;

        AccumulateState state;
        if (pairState.containsKey(pairId)) {
            state = pairState.get(pairId);
        } else {
            state = new AccumulateState();
            state.receivedSubtasks = new BitSet(numSubtasks);
            pairState.put(pairId, state);
            LOG.debug("Received histogram for new pair {}", pairId);
        }

        if (!state.receivedSubtasks.get(sourceSubtaskId)) {
            state.receivedSubtasks.set(sourceSubtaskId);
            state.totalSlices += numSlices;
        }
        state.numReceivedSlices += 1;
        state.histogram =
                null == state.histogram ? histogram : state.histogram.accumulate(histogram);

        if (numSubtasks == state.receivedSubtasks.cardinality()
                && state.numReceivedSlices == state.totalSlices) {
            out.collect(Tuple2.of(pairId, state.histogram));
            LOG.debug("Output accumulated histogram for pair {}", pairId);
            pairState.remove(pairId);
        }
    }

    private static class AccumulateState {
        public BitSet receivedSubtasks = null;
        public int totalSlices = 0;
        public int numReceivedSlices = 0;
        public Histogram histogram = null;
    }
}
