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
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.ps.typeinfo.Long2ObjectOpenHashMapTypeInfo;
import org.apache.flink.ml.common.ps.updater.ModelUpdater;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;

import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/** Base mini batch optimizer. */
public abstract class BaseMiniBatchOptimizer implements ModelUpdater<double[]> {

    protected Long2ObjectOpenHashMap<Tuple2<double[], double[]>> model;

    private ListState<Long2ObjectOpenHashMap<Tuple2<double[], double[]>>> modelDataState;

    private final int modelOffset;
    private final Random random = new Random();
    private final double std;

    public BaseMiniBatchOptimizer(int[] dim, double std) {
        this.modelOffset = dim[1] + dim[2];
        this.std = std;
    }

    @Override
    public double[] get(long[] keys) {
        double[] values = new double[keys.length * modelOffset];
        for (int i = 0; i < keys.length; i++) {
            if (!model.containsKey(keys[i])) {
                double[] sigmaGii = new double[0];
                double[] factors = new double[modelOffset];
                if (keys[i] != -1L) {
                    random.setSeed(keys[i]);
                    for (int j = 0; j < modelOffset; ++j) {
                        factors[j] = std * random.nextDouble();
                    }
                }
                model.put(keys[i], Tuple2.of(factors, sigmaGii));
            }
            System.arraycopy(model.get(keys[i]).f0, 0, values, i * modelOffset, modelOffset);
        }
        return values;
    }

    @Override
    public Iterator<Object> getModelSegments() {
        List<Object> modelSegments = new ArrayList<>();
        for (Long key : model.keySet()) {
            modelSegments.add(Tuple2.of(key, model.get(key.longValue()).f0));
        }
        return modelSegments.iterator();
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        modelDataState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "modelDataState",
                                        new Long2ObjectOpenHashMapTypeInfo<>(
                                                new TupleTypeInfo<>(
                                                        PrimitiveArrayTypeInfo
                                                                .DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO,
                                                        PrimitiveArrayTypeInfo
                                                                .DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO))));
        model =
                OperatorStateUtils.getUniqueElement(modelDataState, "modelDataState")
                        .orElse(new Long2ObjectOpenHashMap<>());
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        modelDataState.clear();
        modelDataState.add(model);
    }
}
