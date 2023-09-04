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
public class AdaDelta extends BaseMiniBatchOptimizer {
    private final int modelOffset;

    private final double gamma;

    public AdaDelta(int[] dim, double std, double gamma) {
        super(dim, std);
        this.modelOffset = dim[1] + dim[2];
        this.gamma = gamma;
    }

    @Override
    public void update(long[] keys, double[] values) {
        for (int i = 0; i < keys.length; ++i) {
            Tuple2<double[], double[]> modelInfo = model.get(keys[i]);
            double[] factors = modelInfo.f0;
            if (modelInfo.f1.length == 0) {
                modelInfo.f1 = new double[2 * modelOffset];
            }

            double[] uOrv = modelInfo.f1;
            int weightIdx = (modelOffset + 1) * (i + 1) - 1;
            if (keys[i] == -1L) {
                double grad = values[(modelOffset + 1) * i] / (values[weightIdx]);
                uOrv[0] = gamma * uOrv[0] + (1 - gamma) * grad * grad;
                double deltaW =
                        -Math.sqrt(uOrv[modelOffset] + EPS) * grad / Math.sqrt(uOrv[0] + EPS);

                factors[0] += deltaW;
                uOrv[modelOffset] = gamma * uOrv[modelOffset] + (1 - gamma) * deltaW * deltaW;
            } else {
                for (int j = 0; j < modelOffset; ++j) {
                    double grad = values[(modelOffset + 1) * i + j] / (values[weightIdx]);
                    uOrv[j] = gamma * uOrv[j] + (1 - gamma) * grad * grad;

                    double deltaW =
                            -Math.sqrt(uOrv[modelOffset + j] + EPS)
                                    * grad
                                    / Math.sqrt(uOrv[j] + EPS);
                    factors[j] += deltaW;
                    uOrv[modelOffset + j] =
                            gamma * uOrv[modelOffset + j] + (1 - gamma) * deltaW * deltaW;
                }
            }
        }
    }
}
