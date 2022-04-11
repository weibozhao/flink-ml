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

package org.apache.flink.ml.util;

import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.util.Preconditions;

/** Utility methods for testing. */
public class TestUtils {
    /** Note: this comparator imposes orderings that are inconsistent with equals. */
    public static int compare(DenseVector first, DenseVector second) {
        Preconditions.checkArgument(first.size() == second.size(), "Vector size mismatched.");
        for (int i = 0; i < first.size(); i++) {
            int cmp = Double.compare(first.get(i), second.get(i));
            if (cmp != 0) {
                return cmp;
            }
        }
        return 0;
    }
}
