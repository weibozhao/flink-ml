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

package org.apache.flink.ml.common.fm.optim.avg;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.fm.BaseAvgOptimizer;
import org.apache.flink.ml.common.fm.BaseFmTrain.LossFunction;
import org.apache.flink.ml.common.fm.FmSample;

import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;

import java.util.List;

import static org.apache.flink.ml.common.fm.BaseFmTrain.EPS;

/**
 * An iteration stage that uses the current batch samples to update the model values with Adam
 * method.
 */
public class Adam extends BaseAvgOptimizer {
    private final LossFunction lossFunc;
    private final int[] dim;
    private final double[] regular;
    private final double beta1;
    private final double beta2;
    private final double learnRate;

    public Adam(
            LossFunction lossFunc,
            int[] dim,
            double[] regular,
            double learnRate,
            double beta1,
            double beta2) {
        super(dim, 3 * (dim[1] + dim[2]) + 3);
        this.lossFunc = lossFunc;
        this.dim = dim;
        this.regular = regular;
        this.learnRate = learnRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
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
                updateOneDim(grad, modelValues, idx, offset, 0);
                modelValues[idx + 3 * offset + 2] += dataPoint.weight;
            }

            for (int i = 0; i < indices.length; ++i) {
                if (indices[i] == -1L) {
                    continue;
                }
                int idx = keyToFactorOffsets.get(indices[i]);
                modelValues[idx + 3 * offset + 2] += dataPoint.weight;

                for (int j = 0; j < dim[2]; j++) {
                    double viXi = vals[i] * modelValues[idx + j];
                    double d = vals[i] * (yVx.f1[j] - viXi);
                    double grad = dLdy * d + regular[2] * modelValues[idx + j];
                    updateOneDim(grad, modelValues, idx, offset, j);
                }
                if (dim[1] > 0) {
                    double grad = dLdy * vals[i] + regular[1] * modelValues[idx + dim[2]];
                    updateOneDim(grad, modelValues, idx, offset, dim[2]);
                }
            }
        }
        return loss;
    }

    private void updateOneDim(double grad, double[] modelValues, int idx, int offset, int bias) {
        if (modelValues[idx + 3 * offset] == 0) {
            modelValues[idx + 3 * offset] = beta1;
            modelValues[idx + 3 * offset + 1] = beta2;
        } else {
            modelValues[idx + 3 * offset] *= beta1;
            modelValues[idx + 3 * offset + 1] *= beta2;
        }

        modelValues[idx + offset + bias] =
                beta1 * modelValues[idx + offset + bias] + (1 - beta1) * grad;
        modelValues[idx + 2 * offset + bias] =
                beta2 * modelValues[idx + 2 * offset + bias] + (1 - beta2) * grad * grad;

        double nBar = modelValues[idx + offset + bias] / (1 - modelValues[idx + 3 * offset]);
        double zBar =
                modelValues[idx + 2 * offset + bias] / (1 - modelValues[idx + 3 * offset + 1]);
        modelValues[idx + bias] -= learnRate * nBar / (Math.sqrt(zBar) + EPS);
    }
}
