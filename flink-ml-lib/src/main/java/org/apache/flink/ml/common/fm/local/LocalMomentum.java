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

package org.apache.flink.ml.common.fm.local;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.fm.BaseFmTrain.LossFunction;
import org.apache.flink.ml.common.fm.BaseLocalOptimizer;
import org.apache.flink.ml.common.fm.FmSample;

import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;

import java.util.List;

/** An iteration stage that uses the pulled model values and batch data to compute the gradients. */
public class LocalMomentum extends BaseLocalOptimizer {
    private final LossFunction lossFunc;
    private final int[] dim;
    private final double[] regular;
    private final double learnRate;
    private final double gamma;

    public LocalMomentum(
            LossFunction lossFunc, int[] dim, double[] regular, double learnRate, double gamma) {
        super(dim, 2 * (dim[1] + dim[2]) + 1);
        this.lossFunc = lossFunc;
        this.dim = dim;
        this.regular = regular;
        this.learnRate = learnRate;
        this.gamma = gamma;
    }

    @Override
    protected double updateFactors(
            List<FmSample> batchData,
            Tuple2<long[], double[]> modelData,
            Long2IntOpenHashMap keyToFactorOffsets) {

        double[] modelValues = modelData.f1;

        int offset = dim[1] + dim[2];

        double loss = 0.0;
        for (FmSample dataPoint : batchData) {
            Tuple2<Double, double[]> yVx =
                    calcY(dataPoint.features, modelValues, dim, keyToFactorOffsets);
            double yTruth = dataPoint.label;
            double dLdy = lossFunc.gradient(yTruth, yVx.f0);
            loss += lossFunc.loss(yTruth, yVx.f0);
            long[] indices = dataPoint.features.f0;
            double[] vals = dataPoint.features.f1;

            if (dim[0] > 0) {
                int idx = keyToFactorOffsets.get(-1L);

                double grad = dLdy + regular[0] * modelValues[idx];
                modelValues[idx + offset] = gamma * modelValues[idx + offset] + learnRate * grad;
                modelValues[idx] -= modelValues[idx + offset];

                modelValues[idx + 2 * offset] += dataPoint.weight;
            }

            for (int i = 0; i < indices.length; ++i) {
                if (indices[i] == -1L) {
                    continue;
                }
                int idx = keyToFactorOffsets.get(indices[i]);
                modelValues[idx + 2 * offset] += dataPoint.weight;

                for (int j = 0; j < dim[2]; j++) {
                    double viXi = vals[i] * modelValues[idx + j];
                    double d = vals[i] * (yVx.f1[j] - viXi);
                    double grad = dLdy * d + regular[2] * modelValues[idx + j];
                    modelValues[idx + offset + j] =
                            gamma * modelValues[idx + offset + j] + learnRate * grad;
                    modelValues[idx + j] -= modelValues[idx + offset + j];
                }
                if (dim[1] > 0) {
                    double grad = dLdy * vals[i] + regular[1] * modelValues[idx + dim[2]];
                    modelValues[idx + offset + dim[2]] =
                            gamma * modelValues[idx + offset + dim[2]] + learnRate * grad;
                    modelValues[idx + dim[2]] -= modelValues[idx + offset + dim[2]];
                }
            }
        }
        return loss;
    }
}
