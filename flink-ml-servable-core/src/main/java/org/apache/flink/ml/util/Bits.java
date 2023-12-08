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

/**
 * Utility methods for packing/unpacking primitive values in/out of byte arrays using big-endian
 * byte ordering. Referenced from java.io.Bits.
 */
public class Bits {

    /*
     * Methods for unpacking primitive values from byte arrays starting at
     * given offsets.
     */

    public static long getLong(byte[] b, int off) {
        return ((b[off + 7] & 0xFFL))
                + ((b[off + 6] & 0xFFL) << 8)
                + ((b[off + 5] & 0xFFL) << 16)
                + ((b[off + 4] & 0xFFL) << 24)
                + ((b[off + 3] & 0xFFL) << 32)
                + ((b[off + 2] & 0xFFL) << 40)
                + ((b[off + 1] & 0xFFL) << 48)
                + (((long) b[off]) << 56);
    }

    public static double getDouble(byte[] b, int off) {
        return Double.longBitsToDouble(getLong(b, off));
    }

    public static float getFloat(byte[] b, int off) {
        return Float.intBitsToFloat(getInt(b, off));
    }

    public static int getInt(byte[] b, int off) {
        return ((b[off + 3] & 0xFF))
                + ((b[off + 2] & 0xFF) << 8)
                + ((b[off + 1] & 0xFF) << 16)
                + ((b[off]) << 24);
    }

    /*
     * Methods for packing primitive values into byte arrays starting at given
     * offsets.
     */

    public static void putLong(byte[] b, int off, long val) {
        b[off + 7] = (byte) (val);
        b[off + 6] = (byte) (val >>> 8);
        b[off + 5] = (byte) (val >>> 16);
        b[off + 4] = (byte) (val >>> 24);
        b[off + 3] = (byte) (val >>> 32);
        b[off + 2] = (byte) (val >>> 40);
        b[off + 1] = (byte) (val >>> 48);
        b[off] = (byte) (val >>> 56);
    }

    public static void putDouble(byte[] b, int off, double val) {
        putLong(b, off, Double.doubleToLongBits(val));
    }

    public static void putFloat(byte[] b, int off, float val) {
        putLong(b, off, Float.floatToIntBits(val));
    }

    public static void putInt(byte[] b, int off, int val) {
        b[off + 3] = (byte) (val);
        b[off + 2] = (byte) (val >>> 8);
        b[off + 1] = (byte) (val >>> 16);
        b[off] = (byte) (val >>> 24);
    }

    /** Gets a long array from the byte array starting from the given offset. */
    public static long[] getLongArray(byte[] bytes, int offset) {
        int size = Bits.getInt(bytes, offset);
        offset += Integer.BYTES;
        long[] result = new long[size];
        for (int i = 0; i < size; i++) {
            result[i] = Bits.getLong(bytes, offset);
            offset += Long.BYTES;
        }
        return result;
    }

    /**
     * Puts a long array to the byte array starting from the given offset.
     *
     * @return the next position to write on.
     */
    public static int putLongArray(long[] array, byte[] bytes, int offset) {
        Bits.putInt(bytes, offset, array.length);
        offset += Integer.BYTES;
        for (int i = 0; i < array.length; i++) {
            Bits.putLong(bytes, offset, array[i]);
            offset += Long.BYTES;
        }
        return offset;
    }

    /** Returns the size of a long array in bytes. */
    public static int getLongArraySizeInBytes(long[] array) {
        return Integer.BYTES + array.length * Long.BYTES;
    }

    /** Returns the size of a long array in bytes. */
    public static int getFloatArraySizeInBytes(float[] array) {
        return Integer.BYTES + array.length * Float.BYTES;
    }

    /** Gets a double array from the byte array starting from the given offset. */
    public static double[] getDoubleArray(byte[] bytes, int offset) {
        int size = Bits.getInt(bytes, offset);
        offset += Integer.BYTES;
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = Bits.getDouble(bytes, offset);
            offset += Long.BYTES;
        }
        return result;
    }

    /**
     * Puts a double array to the byte array starting from the given offset.
     *
     * @return the next position to write on.
     */
    public static int putDoubleArray(double[] array, byte[] bytes, int offset) {
        Bits.putInt(bytes, offset, array.length);
        offset += Integer.BYTES;
        for (int i = 0; i < array.length; i++) {
            Bits.putDouble(bytes, offset, array[i]);
            offset += Double.BYTES;
        }
        return offset;
    }

    /** Gets a double array from the byte array starting from the given offset. */
    public static float[] getFloatArray(byte[] bytes, int offset) {
        int size = Bits.getInt(bytes, offset);
        offset += Integer.BYTES;
        float[] result = new float[size];
        for (int i = 0; i < size; i++) {
            result[i] = Bits.getFloat(bytes, offset);
            offset += Float.BYTES;
        }
        return result;
    }

    /** Returns the size of a double array in bytes. */
    public static int getDoubleArraySizeInBytes(double[] array) {
        return Integer.BYTES + array.length * Long.BYTES;
    }

    public static int putFloatArray(float[] array, byte[] bytes, int offset) {
        Bits.putInt(bytes, offset, array.length);
        offset += Integer.BYTES;
        for (float v : array) {
            Bits.putInt(bytes, offset, Float.floatToIntBits(v));
            offset += Float.BYTES;
        }
        return offset;
    }
}
