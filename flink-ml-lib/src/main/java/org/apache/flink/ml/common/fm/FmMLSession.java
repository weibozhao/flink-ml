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

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.ml.common.ps.iterations.MLSessionImpl;
import org.apache.flink.ml.common.ps.sarray.SharedDoubleArray;
import org.apache.flink.ml.common.ps.sarray.SharedLongArray;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.util.OutputTag;

import org.apache.commons.collections.IteratorUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/** The ML session for machine learning algorithms that adopts fm model training. */
public class FmMLSession extends MLSessionImpl<FmSample> {

    /** The placeholder for indices to push/pull for each iteration. */
    public final SharedLongArray indices;
    /** The placeholder for the push/pull values for each iteration. */
    public final SharedDoubleArray values;
    /** The placeholder for local loss of the last batch. */
    public Double[] localLoss;
    /** The placeholder for global loss of the last evaluation. */
    public Double[] globalLoss;

    private boolean isInitialized = false;

    private ListState<Double> lossState;

    /** The batch of training data for computing gradients. */
    public List<FmSample> batchData;

    private ListState<FmSample> batchDataState;
    /** Global batch size. */
    private final int globalBatchSize;
    /** The local batch size. */
    private int localBatchSize;

    public FmMLSession(int globalBatchSize) {
        this(globalBatchSize, null);
    }

    public FmMLSession(int globalBatchSize, List<OutputTag<?>> outputTags) {
        super(outputTags);
        this.globalBatchSize = globalBatchSize;
        this.indices = new SharedLongArray();
        this.values = new SharedDoubleArray();
    }

    @Override
    public void setWorldInfo(int workerId, int numWorkers) {
        super.setWorldInfo(workerId, numWorkers);
        this.localBatchSize = globalBatchSize / numWorkers;
        if (globalBatchSize % numWorkers > workerId) {
            localBatchSize++;
        }
        this.batchData = new ArrayList<>(localBatchSize);
    }

    @Override
    @SuppressWarnings("unchecked")
    public void initializeState(StateInitializationContext context) throws Exception {
        batchDataState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "batchDataState", TypeInformation.of(FmSample.class)));

        Iterator<FmSample> batchDataIterator = batchDataState.get().iterator();
        if (batchDataIterator.hasNext()) {
            batchData = IteratorUtils.toList(batchDataIterator);
        }

        lossState =
                context.getOperatorStateStore()
                        .getListState(new ListStateDescriptor<>("lossState", Types.DOUBLE));
        Iterator<Double> initLoss = lossState.get().iterator();
        if (initLoss.hasNext()) {
            localLoss = new Double[] {initLoss.next(), initLoss.next()};
            globalLoss = new Double[] {initLoss.next(), initLoss.next()};
        } else {
            localLoss = new Double[] {Double.MAX_VALUE, 0.0, 0.0};
            globalLoss = new Double[] {Double.MAX_VALUE, .0, 0.0};
        }
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        batchDataState.clear();
        if (batchData.size() > 0) {
            batchDataState.addAll(batchData);
        }
        lossState.clear();
        lossState.add(localLoss[0]);
        lossState.add(localLoss[1]);
        lossState.add(globalLoss[0]);
        lossState.add(globalLoss[1]);
    }

    /** Reads in next batch of training data. */
    public void readInNextBatchData() throws IOException {
        if (!isInitialized) {
            int cnt = 0;
            while (inputData.hasNext()) {
                inputData.next();
                cnt++;
            }
            localLoss[2] =
                    Math.max(1, cnt / localBatchSize + (cnt % localBatchSize == 0 ? 0 : 1)) * 1.;
            inputData.reset();
            isInitialized = true;
        }
        batchData.clear();
        int i = 0;
        while (i < localBatchSize && inputData.hasNext()) {
            batchData.add(inputData.next());
            i++;
        }
        if (!inputData.hasNext()) {
            inputData.reset();
        }
    }
}
