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

package org.apache.flink.ml.common.ps.training;

import java.util.function.Supplier;

/** A communication stage that push (indices, values) to servers. */
public class PushStage implements IterationStage {
    public final Supplier<long[]> keysSupplier;
    public final Supplier<double[]> valuesSupplier;

    public PushStage(Supplier<long[]> keysSupplier, Supplier<double[]> valuesSupplier) {
        this.keysSupplier = keysSupplier;
        this.valuesSupplier = valuesSupplier;
    }
}
