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

package org.apache.flink.ml.regression.fmregressor;

import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.fm.FmModelData;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.List;

class GenerateModelData implements MapPartitionFunction<Object, FmModelData> {

    private final List<Tuple2<Long, float[]>> factors = new ArrayList<>();
    private final int[] dim;
    private final boolean isReg;

    public GenerateModelData(int[] dim, boolean isReg) {
        this.dim = dim;
        this.isReg = isReg;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void mapPartition(Iterable<Object> iterable, Collector<FmModelData> collector) {
        for (Object ele : iterable) {
            Tuple2<Long, double[]> t2 = (Tuple2<Long, double[]>) ele;
            float[] factor = new float[t2.f1.length];
            for (int i = 0; i < factor.length; ++i) {
                factor[i] = (float) t2.f1[i];
            }
            factors.add(Tuple2.of(t2.f0, factor));
        }
        collector.collect(new FmModelData(factors, dim, isReg));
    }
}
