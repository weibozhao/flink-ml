/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.flink.ml.algo.batch.knn;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.table.types.DataType;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import java.util.List;
import java.util.PriorityQueue;

/** Utility to operator to interact with Table contents, such as rows and columns. */
public class KnnUtils {

    /**
     * Find the index of <code>targetCol</code> in string array <code>tableCols</code>. It will
     * ignore the case of the tableCols.
     *
     * @param tableCols a string array among which to find the targetCol.
     * @param targetCol the targetCol to find.
     * @return the index of the targetCol, if not found, returns -1.
     */
    public static int findColIndex(String[] tableCols, String targetCol) {
        Preconditions.checkNotNull(targetCol, "targetCol is null!");
        for (int i = 0; i < tableCols.length; i++) {
            if (targetCol.equalsIgnoreCase(tableCols[i])) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Find the indices of <code>targetCols</code> in string array <code>tableCols</code>. If <code>
     *     targetCols
     * </code> is null, it will be replaced by the <code>tableCols</code>
     *
     * @param tableCols a string array among which to find the targetCols.
     * @param targetCols the targetCols to find.
     * @return the indices of the targetCols.
     */
    public static int[] findColIndices(String[] tableCols, String[] targetCols) {
        if (targetCols == null) {
            int[] indices = new int[tableCols.length];
            for (int i = 0; i < tableCols.length; i++) {
                indices[i] = i;
            }
            return indices;
        }
        int[] indices = new int[targetCols.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = findColIndex(tableCols, targetCols[i]);
        }
        return indices;
    }

    /**
     * transform resolvedSchema to schema.
     *
     * @param resolvedSchema input resolvedSchema.
     * @return output schema.
     */
    public static Schema resolvedSchema2Schema(ResolvedSchema resolvedSchema) {
        Schema.Builder builder = Schema.newBuilder();
        List<String> colNames = resolvedSchema.getColumnNames();
        List<DataType> colTypes = resolvedSchema.getColumnDataTypes();
        for (int i = 0; i < colNames.size(); ++i) {
            builder.column(colNames.get(i), colTypes.get(i).getLogicalType().toString());
        }
        return builder.build();
    }

    /**
     * find column type.
     *
     * @param tableSchema schema.
     * @param targetCols column names.
     * @return column types.
     */
    public static DataType[] findColTypes(ResolvedSchema tableSchema, String[] targetCols) {
        if (targetCols == null) {
            return tableSchema.getColumnDataTypes().toArray(new DataType[0]);
        }
        DataType[] types = new DataType[targetCols.length];
        String[] colNames = tableSchema.getColumnNames().toArray(new String[0]);
        for (int i = 0; i < types.length; i++) {
            types[i] = tableSchema.getColumnDataTypes().get(findColIndex(colNames, targetCols[i]));
        }
        return types;
    }

    /**
     * cast data x to t type.
     *
     * @param x data.
     * @param t type.
     * @return
     */
    public static Object castTo(Object x, DataType t) {
        if (x == null) {
            return null;
        } else if (t.equals(DataTypes.BOOLEAN())) {
            if (x instanceof Boolean) {
                return x;
            }
            return Boolean.valueOf(x.toString());
        } else if (t.equals(DataTypes.BYTES())) {
            if (x instanceof Number) {
                return ((Number) x).byteValue();
            }
            return Byte.valueOf(x.toString());
        } else if (t.equals(DataTypes.INT())) {
            if (x instanceof Number) {
                return ((Number) x).intValue();
            }
            return Integer.valueOf(x.toString());
        } else if (t.equals(DataTypes.BIGINT())) {
            if (x instanceof Number) {
                return ((Number) x).longValue();
            }
            return Long.valueOf(x.toString());
        } else if (t.equals(DataTypes.FLOAT())) {
            if (x instanceof Number) {
                return ((Number) x).floatValue();
            }
            return Float.valueOf(x.toString());
        } else if (t.equals(DataTypes.DOUBLE())) {
            if (x instanceof Number) {
                return ((Number) x).doubleValue();
            }
            return Double.valueOf(x.toString());
        } else if (t.equals(DataTypes.STRING())) {
            if (x instanceof String) {
                return x;
            }
            return x.toString();
        } else {
            throw new RuntimeException("unsupported type: " + t.getClass().getName());
        }
    }

    /**
     * update queue.
     *
     * @param map queue.
     * @param topN top N.
     * @param newValue new value.
     * @param head head value.
     * @param <T> id type.
     * @return head value.
     */
    public static <T> Tuple2<Double, T> updateQueue(
            PriorityQueue<Tuple2<Double, T>> map,
            int topN,
            Tuple2<Double, T> newValue,
            Tuple2<Double, T> head) {
        if (null == newValue) {
            return head;
        }
        if (map.size() < topN) {
            map.add(Tuple2.of(newValue.f0, newValue.f1));
            head = map.peek();
        } else {
            if (map.comparator().compare(head, newValue) < 0) {
                Tuple2<Double, T> peek = map.poll();
                peek.f0 = newValue.f0;
                peek.f1 = newValue.f1;
                map.add(peek);
                head = map.peek();
            }
        }
        return head;
    }

    /**
     * merge two rows to one.
     *
     * @param rec1 row 1.
     * @param rec2 row 2.
     * @return new row.
     */
    public static Row merge(Row rec1, Row rec2) {
        int n1 = rec1.getArity();
        int n2 = rec2.getArity();
        Row ret = new Row(n1 + n2);
        for (int i = 0; i < n1; ++i) {
            ret.setField(i, rec1.getField(i));
        }
        for (int i = 0; i < n2; ++i) {
            ret.setField(i + n1, rec2.getField(i));
        }
        return ret;
    }
}
