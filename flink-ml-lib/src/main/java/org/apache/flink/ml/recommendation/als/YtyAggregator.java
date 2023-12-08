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

package org.apache.flink.ml.recommendation.als;

import org.apache.flink.ml.common.ps.iterations.PullComponent.Aggregator;

/** Comments. */
class YtyAggregator implements Aggregator<float[], float[]> {
    @Override
    public float[] add(float[] in, float[] acc) {

        if (acc == null) {
            acc = new float[in.length * in.length];
        }
        calcYty(in, acc);
        return acc;
    }

    @Override
    public float[] merge(float[] acc1, float[] acc2) {
        for (int i = 0; i < acc1.length; i++) {
            acc2[i] += acc1[i];
        }
        return acc2;
    }

    private void calcYty(float[] vec, float[] result) {
        for (int i = 0; i < vec.length; i++) {
            for (int j = 0; j < vec.length; j++) {
                result[i * vec.length + j] += vec[i] * vec[j];
            }
        }
    }
}
