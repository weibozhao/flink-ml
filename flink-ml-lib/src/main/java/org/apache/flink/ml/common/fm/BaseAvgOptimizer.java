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
import org.apache.flink.ml.common.ps.iterations.ProcessComponent;
import org.apache.flink.ml.common.ps.sarray.SharedDoubleArray;
import org.apache.flink.ml.common.ps.sarray.SharedLongArray;

import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;

import java.io.IOException;
import java.util.List;

/**
 * An iteration stage that uses the current batch samples to update the pulled model values with
 * special method. This is a abstract class, the derived class must implement the updateFactors()
 * function.
 */
public abstract class BaseAvgOptimizer extends ProcessComponent<FmMLSession> {
    private final int[] dim;
    private final int modelOffset;

    public BaseAvgOptimizer(int[] dim, int modelOffset) {
        this.dim = dim;
        this.modelOffset = modelOffset;
    }

    @Override
    public void process(FmMLSession session) throws IOException {
        SharedLongArray modelIndices = session.indices;
        SharedDoubleArray modelValues = session.values;
        int idxSize = session.indices.size();
        Long2IntOpenHashMap keyToFactorOffsets = new Long2IntOpenHashMap(idxSize);
        int offset = dim[1] + dim[2];
        for (int i = 0; i < idxSize; ++i) {
            keyToFactorOffsets.put(modelIndices.elements()[i], modelOffset * i);
        }
        double oldLoss = session.localLoss[0];
        double loss;
        int cnt = 1;
        while (true) {
            loss =
                    updateFactors(
                            session.batchData,
                            Tuple2.of(modelIndices.elements(), modelValues.elements()),
                            keyToFactorOffsets);
            if (oldLoss - loss < 1.0e-3 * oldLoss || cnt > 4) {
                break;
            }
            if (cnt == 1) {
                session.localLoss[0] = loss;
                session.localLoss[1] = session.batchData.size() * 1.;
            }
            cnt++;
        }

        for (int i = 0; i < modelIndices.size(); ++i) {
            int numOffset = modelValues.size() / modelIndices.size() / offset;
            int bias = (modelValues.size() / modelIndices.size()) % offset - 1;
            double weight = modelValues.elements()[i * modelOffset + numOffset * offset + bias];
            for (int j = 0; j < numOffset * offset; ++j) {
                modelValues.elements()[i * modelOffset + j] *= weight;
            }
        }
    }

    /**
     * Updates the model data with given samples in current batchData.
     *
     * @param batchData The current batch data
     * @param modelData The model data to be updated
     * @param keyToFactorOffsets The Map for key to factor position
     * @return The loss of current batch data
     */
    protected abstract double updateFactors(
            List<FmSample> batchData,
            Tuple2<long[], double[]> modelData,
            Long2IntOpenHashMap keyToFactorOffsets);

    /** calculate the value of y with given fm model. */
    protected Tuple2<Double, double[]> calcY(
            Tuple2<long[], double[]> feature,
            double[] modelData,
            int[] dim,
            Long2IntOpenHashMap keyToModelOffsets) {

        long[] featureIds = feature.f0;
        double[] featureValues = feature.f1;

        double[] vx = new double[dim[2]];
        double[] v2x2 = new double[dim[2]];

        // (1) compute y
        double y = 0.;

        if (dim[0] > 0) {
            y += modelData[keyToModelOffsets.get(-1L)];
        }

        for (int i = 0; i < featureIds.length; i++) {
            if (featureIds[i] == -1L) {
                continue;
            }
            int featurePos = keyToModelOffsets.get(featureIds[i]);
            double x = featureValues[i];

            // the linear term
            if (dim[1] > 0) {
                y += x * modelData[featurePos + dim[2]];
            }
            // the quadratic term
            for (int j = 0; j < dim[2]; j++) {
                double viXi = x * modelData[featurePos + j];
                vx[j] += viXi;
                v2x2[j] += viXi * viXi;
            }
        }

        for (int i = 0; i < dim[2]; i++) {
            y += 0.5 * (vx[i] * vx[i] - v2x2[i]);
        }
        return Tuple2.of(y, vx);
    }
}
