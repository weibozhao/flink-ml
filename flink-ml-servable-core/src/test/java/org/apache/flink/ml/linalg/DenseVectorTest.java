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

package org.apache.flink.ml.linalg;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests the behavior of {@link DenseIntDoubleVector}. */
public class DenseVectorTest {

    private static final double TOLERANCE = 1e-7;

    @Test
    public void testClone() {
        DenseIntDoubleVector denseVec = Vectors.dense(1, 2, 3);
        DenseIntDoubleVector clonedDenseVec = denseVec.clone();
        assertArrayEquals(clonedDenseVec.values, new double[] {1, 2, 3}, TOLERANCE);

        clonedDenseVec.values[0] = -1;
        assertArrayEquals(denseVec.values, new double[] {1, 2, 3}, TOLERANCE);
        assertArrayEquals(clonedDenseVec.values, new double[] {-1, 2, 3}, TOLERANCE);
    }

    @Test
    public void testGetAndSet() {
        DenseIntDoubleVector denseVec = Vectors.dense(1, 2, 3);
        assertEquals(1, denseVec.get(0), TOLERANCE);

        denseVec.set(0, 2.0);
        assertEquals(2, denseVec.get(0), TOLERANCE);
    }
}
