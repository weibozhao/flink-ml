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

package org.apache.flink.ml.common.ps.iterations;

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.ml.common.ps.sarray.SharedDoubleArray;
import org.apache.flink.ml.common.ps.sarray.SharedFloatArray;
import org.apache.flink.ml.common.ps.sarray.SharedLongArray;
import org.apache.flink.ml.common.ps.sarray.SharedNumericalArray;
import org.apache.flink.util.function.SerializableSupplier;

import java.util.function.Supplier;

/**
 * An iteration stage that push (indices, values) to servers. User can specify how values from
 * different workers are merged via {@code PushStage#reduceFunc}. By default, the values are summed
 * from different workers.
 *
 * <p>Note that the length of the values array must be divisible by the length of the keys array.
 * Additionally, each value corresponding to a given key must have the same length. For instance,
 * considering the keys {1, 4} and values {1,2,3,4,5,6}, the value at index 1 would be {1,2,3}, and
 * the value at index 4 would be {4,5,6}.
 */
public class PushComponent implements PsIterationComponent {
    public final Supplier<SharedLongArray> keys;
    public final Supplier<SharedNumericalArray> values;

    /** The function to reduce the pushes from all workers. For gradient descent based methods, */
    public ReduceFunction<Double> reduceFunc;

    public ReduceFunction<Float> floatReduceFunc;

    public PushComponent(
            SerializableSupplier<SharedLongArray> keys,
            SerializableSupplier<SharedNumericalArray> values) {
        this.keys = keys;
        this.values = values;

        Object obj = values.get();
        if (obj instanceof SharedDoubleArray) {
            reduceFunc = Double::sum;
        } else if (obj instanceof SharedFloatArray) {
            floatReduceFunc = Float::sum;
        }
    }

    public PushComponent(
            SerializableSupplier<SharedLongArray> keys,
            SerializableSupplier<SharedNumericalArray> values,
            ReduceFunction reduceFunction,
            boolean isDouble) {
        this.keys = keys;
        this.values = values;
        if (isDouble) {
            this.reduceFunc = (ReduceFunction<Double>) reduceFunction;
        } else {
            this.floatReduceFunc = (ReduceFunction<Float>) reduceFunction;
        }
    }
}
