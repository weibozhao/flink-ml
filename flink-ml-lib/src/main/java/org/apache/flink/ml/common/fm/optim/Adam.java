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

package org.apache.flink.ml.common.fm.optim;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.fm.BaseMiniBatchOptimizer;

import static org.apache.flink.ml.common.fm.BaseFmTrain.EPS;

/** Adam optimizer. */
public class Adam extends BaseMiniBatchOptimizer {
    private final int modelOffset;
    private final double learnRate;

    private final double beta1;
    private final double beta2;

    public Adam(int[] dim, double learnRate, double std, double beta1, double beta2) {
        super(dim, std);
        this.modelOffset = dim[1] + dim[2];
        this.learnRate = learnRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    @Override
    public void update(long[] keys, double[] values) {
        for (int i = 0; i < keys.length; ++i) {
            Tuple2<double[], double[]> modelInfo = model.get(keys[i]);
            double[] factors = modelInfo.f0;
            if (modelInfo.f1.length == 0) {
                modelInfo.f1 = new double[2 * modelOffset + 2];
                modelInfo.f1[2 * modelOffset] = 1.0;
                modelInfo.f1[2 * modelOffset + 1] = 1.0;
            }
            modelInfo.f1[2 * modelOffset] *= beta1;
            modelInfo.f1[2 * modelOffset + 1] *= beta2;

            double beta1Power = modelInfo.f1[2 * modelOffset];
            double beta2Power = modelInfo.f1[2 * modelOffset + 1];
            double[] mOrn = modelInfo.f1;
            int weightIdx = (modelOffset + 1) * (i + 1) - 1;
            if (keys[i] == -1L) {
                double grad = values[(modelOffset + 1) * i] / (values[weightIdx]);
                mOrn[0] = beta1 * mOrn[0] + (1 - beta1) * grad;
                mOrn[modelOffset] = beta2 * mOrn[modelOffset] + (1 - beta2) * grad * grad;

                double nBar = mOrn[0] / (1 - beta1Power);
                double zBar = mOrn[modelOffset] / (1 - beta2Power);
                factors[0] -= learnRate * nBar / (Math.sqrt(zBar) + EPS);
            } else {
                for (int j = 0; j < modelOffset; ++j) {
                    double grad = values[(modelOffset + 1) * i + j] / (values[weightIdx]);
                    mOrn[j] = beta1 * mOrn[j] + (1 - beta1) * grad;
                    mOrn[modelOffset + j] =
                            beta2 * mOrn[modelOffset + j] + (1 - beta2) * grad * grad;
                    double nBar = mOrn[j] / (1 - beta1Power);
                    double zBar = mOrn[modelOffset + j] / (1 - beta2Power);
                    factors[j] -= learnRate * nBar / (Math.sqrt(zBar) + EPS);
                }
            }
        }
    }
}
