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

package org.apache.flink.ml.feature.textdedup.similarity;

import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;

import org.apache.flink.shaded.guava30.com.google.common.hash.HashFunction;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.Random;

import static org.apache.flink.shaded.guava30.com.google.common.hash.Hashing.murmur3_128;

class MinHashFunction {
    protected static final int HASH_PRIME = 2038074743;
    protected final IntArrayHasher intArrayHasher;
    protected int[][] randCoefficientsA;
    protected int[][] randCoefficientsB;

    public MinHashFunction(long seed, int numProjectionsPerTable, int numHashTables) {
        Random random = new Random(seed);
        randCoefficientsA = new int[numHashTables][numProjectionsPerTable];
        randCoefficientsB = new int[numHashTables][numProjectionsPerTable];
        for (int i = 0; i < numHashTables; i++) {
            for (int j = 0; j < numProjectionsPerTable; j++) {
                randCoefficientsA[i][j] = 1 + random.nextInt(HASH_PRIME - 1);
                randCoefficientsB[i][j] = random.nextInt(HASH_PRIME - 1);
            }
        }
        intArrayHasher = new IntArrayHasher(numProjectionsPerTable);
    }

    /**
     * Hashes a vector to a long array of length `numHashTables`.
     *
     * @param vec The vector to be hashed.
     * @return The hash results.
     */
    public long[] hashFunctionToLong(Vector vec) {
        int numHashTables = randCoefficientsA.length;
        long[] minHashSet = new long[numHashTables];
        int[] hashValues = new int[randCoefficientsA[0].length];
        for (int i = 0; i < numHashTables; i += 1) {
            computeHashTable(hashValues, vec, i);
            minHashSet[i] = intArrayHasher.hashToLong(hashValues);
        }
        return minHashSet;
    }

    /**
     * Computes a hash table for a vector with `numProjectionsPerTable` hash functions.
     *
     * @param hashValues The array to store hash results.
     * @param vec The vector to be hashed.
     * @param i The index of hash table.
     */
    public void computeHashTable(int[] hashValues, Vector vec, int i) {
        if (vec instanceof DenseVector) {
            computeHashTable(hashValues, (DenseVector) vec, i);
        } else {
            computeHashTable(hashValues, (SparseVector) vec, i);
        }
    }

    public void computeHashTable(int[] hashValues, DenseVector dv, int i) {
        double[] elem = dv.values;
        for (int j = 0; j < hashValues.length; j++) {
            int tmp = HASH_PRIME, cur;
            for (int m = 0; m < elem.length; m++) {
                if (elem[m] != 0) {
                    cur =
                            (int)
                                    (((1L + m) * randCoefficientsA[i][j] + randCoefficientsB[i][j])
                                            % HASH_PRIME);
                    tmp = Math.min(tmp, cur);
                }
            }
            hashValues[j] = tmp;
        }
    }

    public void computeHashTable(int[] hashValues, SparseVector sv, int i) {
        int[] indices = sv.indices;
        for (int j = 0; j < hashValues.length; j++) {
            int tmp = HASH_PRIME, cur;
            for (int index : indices) {
                cur =
                        (int)
                                (((1L + index) * randCoefficientsA[i][j] + randCoefficientsB[i][j])
                                        % HASH_PRIME);
                tmp = Math.min(tmp, cur);
            }
            hashValues[j] = tmp;
        }
    }

    static class IntArrayHasher {
        private static final HashFunction HASH = murmur3_128(0);
        private final ByteBuffer byteBuffer;
        private final IntBuffer intBuffer;

        public IntArrayHasher(int len) {
            byteBuffer = ByteBuffer.allocate(Integer.BYTES * len);
            intBuffer = byteBuffer.asIntBuffer();
        }

        public long hashToLong(int[] arr) {
            intBuffer.rewind();
            intBuffer.put(arr);
            return HASH.hashBytes(byteBuffer.array()).asLong();
        }
    }
}
