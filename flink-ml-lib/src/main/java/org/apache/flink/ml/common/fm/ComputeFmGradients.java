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
import org.apache.flink.ml.common.fm.BaseFmTrain.LossFunction;
import org.apache.flink.ml.common.ps.iterations.ProcessComponent;
import org.apache.flink.ml.common.ps.sarray.SharedDoubleArray;
import org.apache.flink.ml.common.ps.sarray.SharedLongArray;

import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;

import java.io.IOException;
import java.util.List;

/** An iteration stage that uses the pulled model values and batch data to compute the gradients. */
public class ComputeFmGradients extends ProcessComponent<FmMLSession> {
    private final LossFunction lossFunc;
    private final int[] dim;
    private final double[] regular;

    public ComputeFmGradients(LossFunction lossFunc, int[] dim, double[] regular) {
        this.lossFunc = lossFunc;
        this.dim = dim;
        this.regular = regular;
    }

    @Override
    public void process(FmMLSession session) throws IOException {
        SharedLongArray modelIndices = session.indices;
        SharedDoubleArray modelValues = session.values;
        double[] gradients =
                computeGradient(
                        session.batchData,
                        Tuple2.of(modelIndices.elements(), modelValues.elements()),
                        session.indices.size(),
                        session);
        session.values.clear();
        session.values.addAll(gradients);
    }

    private double[] computeGradient(
            List<FmSample> batchData,
            Tuple2<long[], double[]> modelData,
            int idxSize,
            FmMLSession session) {
        int offset = dim[1] + dim[2];
        long[] modelIndices = modelData.f0;
        double[] modelValues = modelData.f1;
        double[] gradients = new double[idxSize * (offset + 1)];

        Long2IntOpenHashMap keyToModelOffsets = new Long2IntOpenHashMap(idxSize);
        Long2IntOpenHashMap keyToGradOffsets = new Long2IntOpenHashMap(idxSize);
        for (int i = 0; i < idxSize; ++i) {
            keyToModelOffsets.put(modelIndices[i], offset * i);
            keyToGradOffsets.put(modelIndices[i], (offset + 1) * i);
        }
        double loss = 0.0;
        for (FmSample dataPoint : batchData) {

            Tuple2<Double, double[]> yVx =
                    calcY(dataPoint.features, modelValues, dim, keyToModelOffsets);
            double yTruth = dataPoint.label;
            double grad = lossFunc.gradient(yTruth, yVx.f0);
            loss += lossFunc.loss(yTruth, yVx.f0);
            long[] indices = dataPoint.features.f0;
            double[] vals = dataPoint.features.f1;

            if (dim[0] > 0) {
                gradients[keyToGradOffsets.get(-1L)] +=
                        grad + regular[0] * modelValues[keyToModelOffsets.get(-1L)];
                gradients[keyToGradOffsets.get(-1L) + offset] += dataPoint.weight;
            }

            for (int i = 0; i < indices.length; ++i) {
                if (indices[i] == -1L) {
                    continue;
                }
                int modelIdx = keyToModelOffsets.get(indices[i]);
                int gradientIdx = keyToGradOffsets.get(indices[i]);
                gradients[gradientIdx + offset] += dataPoint.weight;

                for (int j = 0; j < dim[2]; j++) {
                    double viXi = vals[i] * modelValues[modelIdx + j];
                    double d = vals[i] * (yVx.f1[j] - viXi);
                    gradients[gradientIdx + j] += grad * d + regular[2] * modelValues[modelIdx + j];
                }
                if (dim[1] > 0) {
                    gradients[gradientIdx + dim[2]] +=
                            grad * vals[i] + regular[1] * modelValues[modelIdx + dim[2]];
                }
            }
        }
        session.localLoss[0] = loss;
        session.localLoss[1] = batchData.size() * 1.;
        return gradients;
    }

    /** calculate the value of y with given fm model. */
    private Tuple2<Double, double[]> calcY(
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
