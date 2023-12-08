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

/** RMSProp optimizer. */
public class Momentum extends BaseMiniBatchOptimizer {
    private final int modelOffset;
    private final double learnRate;
    private final double gamma;

    public Momentum(int[] dim, double learnRate, double std, double gamma) {
        super(dim, std);
        this.modelOffset = dim[1] + dim[2];
        this.learnRate = learnRate;
        this.gamma = gamma;
    }

    @Override
    public void update(long[] keys, double[] values) {
        for (int i = 0; i < keys.length; ++i) {
            Tuple2<double[], double[]> modelInfo = model.get(keys[i]);
            double[] factors = modelInfo.f0;
            if (modelInfo.f1.length == 0) {
                modelInfo.f1 = new double[modelOffset];
            }
            double[] deltaW = modelInfo.f1;

            int weightIdx = (modelOffset + 1) * (i + 1) - 1;
            if (keys[i] == -1L) {
                double grad = values[(modelOffset + 1) * i] / (values[weightIdx]);
                deltaW[0] = gamma * deltaW[0] + learnRate * grad;
                factors[0] -= deltaW[0];
            } else {
                for (int j = 0; j < modelOffset; ++j) {
                    double grad = values[(modelOffset + 1) * i + j] / (values[weightIdx]);
                    deltaW[j] = gamma * deltaW[j] + learnRate * grad;
                    factors[j] -= deltaW[j];
                }
            }
        }
    }
}
