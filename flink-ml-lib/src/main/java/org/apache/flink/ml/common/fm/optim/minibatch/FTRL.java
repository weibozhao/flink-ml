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

package org.apache.flink.ml.common.fm.optim.minibatch;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.fm.BaseMiniBatchOptimizer;

/** FTRL optimizer. */
public class FTRL extends BaseMiniBatchOptimizer {
    private final int modelOffset;
    private final double alpha;
    private final double beta;
    private final double l1;
    private final double l2;

    public FTRL(int[] dim, double std, double l1, double l2, double alpha, double beta) {
        super(dim, std);
        this.modelOffset = dim[1] + dim[2];
        this.alpha = alpha;
        this.beta = beta;
        this.l1 = l1;
        this.l2 = l2;
    }

    @Override
    public void update(long[] keys, double[] values) {
        for (int i = 0; i < keys.length; ++i) {
            Tuple2<double[], double[]> modelInfo = model.get(keys[i]);
            double[] factors = modelInfo.f0;
            if (modelInfo.f1.length == 0) {
                modelInfo.f1 = new double[2 * modelOffset];
            }
            double[] nOrz = modelInfo.f1;

            int weightIdx = (modelOffset + 1) * (i + 1) - 1;
            if (keys[i] == -1L) {
                double grad = values[(modelOffset + 1) * i] / (values[weightIdx]);
                double sigma = (Math.sqrt(nOrz[0] + grad * grad) - Math.sqrt(nOrz[0])) / alpha;
                nOrz[modelOffset] += grad - sigma * factors[0];
                nOrz[0] += grad * grad;
                if (Math.abs(nOrz[modelOffset]) <= l1) {
                    factors[0] = 0.0;
                } else {
                    factors[0] =
                            ((nOrz[modelOffset] < 0 ? -1 : 1) * l1 - nOrz[modelOffset])
                                    / ((beta + Math.sqrt(nOrz[0])) / alpha + l2);
                }
            } else {
                for (int j = 0; j < modelOffset; ++j) {
                    double grad = values[(modelOffset + 1) * i + j] / (values[weightIdx]);

                    double sigma = (Math.sqrt(nOrz[j] + grad * grad) - Math.sqrt(nOrz[j])) / alpha;
                    nOrz[modelOffset + j] += grad - sigma * factors[j];
                    nOrz[j] += grad * grad;
                    if (Math.abs(nOrz[modelOffset + j]) <= l1) {
                        factors[j] = 0.0;
                    } else {
                        factors[j] =
                                ((nOrz[modelOffset + j] < 0 ? -1 : 1) * l1 - nOrz[modelOffset + j])
                                        / ((beta + Math.sqrt(nOrz[j])) / alpha + l2);
                    }
                }
            }
        }
    }
}
