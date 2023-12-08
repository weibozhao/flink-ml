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

package org.apache.flink.ml.common.ps.sarray;

import it.unimi.dsi.fastutil.floats.FloatArrayList;

/** A resizable float array that can be shared among different iterations for memory efficiency. */
public class SharedFloatArray implements SharedNumericalArray {

    /** The underlying FloatArrayList that holds the elements. */
    private final FloatArrayList floats;

    /**
     * Constructs a new SDArray from the given float array.
     *
     * @param array the float array to wrap
     */
    public SharedFloatArray(float[] array) {
        floats = FloatArrayList.wrap(array);
    }

    /**
     * Constructs a new SDArray with the given initial capacity.
     *
     * @param capacity the initial capacity
     */
    public SharedFloatArray(int capacity) {
        floats = new FloatArrayList(capacity);
    }

    /** Constructs a new empty SDArray. */
    public SharedFloatArray() {
        floats = new FloatArrayList();
    }

    /**
     * Returns the element at the specified index.
     *
     * @param index the index of the element to return
     * @return the element at the specified index
     */
    public float get(int index) {
        return floats.getFloat(index);
    }

    /**
     * Appends the specified element to the end of this array.
     *
     * @param v the element to add
     */
    public void add(float v) {
        floats.add(v);
    }

    /**
     * Appends all the elements from the specified float array to the end of this array.
     *
     * @param src the float array to append
     */
    public void addAll(float[] src) {
        int sizeBefore = size();
        floats.size(sizeBefore + src.length);
        System.arraycopy(src, 0, elements(), sizeBefore, src.length);
    }

    /**
     * Returns the number of valid elements in this array.
     *
     * @return the number of valid elements in this array
     */
    public int size() {
        return floats.size();
    }

    /**
     * Sets the size of the array to the provided size. If the new size is larger than the current
     * size, the new allocated memory are filled with zero.
     *
     * @param size the new size of the array
     */
    public void size(int size) {
        floats.size(size);
    }

    /** Clears the elements in this array. Note that the memory is not recycled. */
    public void clear() {
        floats.clear();
    }

    /**
     * Returns a float array containing all the elements in this array. Only the first {@link
     * SharedFloatArray#size()} elements are valid.
     *
     * @return a float array containing the all the elements in this array
     */
    public float[] elements() {
        return floats.elements();
    }
}
