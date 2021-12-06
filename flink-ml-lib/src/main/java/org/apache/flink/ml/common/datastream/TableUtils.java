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

package org.apache.flink.ml.common.datastream;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Schema.UnresolvedPhysicalColumn;
import org.apache.flink.table.catalog.Column;
import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.table.runtime.typeutils.ExternalTypeInfo;
import org.apache.flink.table.types.DataType;

import org.apache.commons.lang3.ArrayUtils;

/** Utility class for table-related operations. */
public class TableUtils {
    // Constructs a RowTypeInfo from the given schema.
    public static RowTypeInfo getRowTypeInfo(ResolvedSchema schema) {
        TypeInformation<?>[] types = new TypeInformation<?>[schema.getColumnCount()];
        String[] names = new String[schema.getColumnCount()];

        for (int i = 0; i < schema.getColumnCount(); i++) {
            Column column = schema.getColumn(i).get();
            types[i] = ExternalTypeInfo.of(column.getDataType());
            names[i] = column.getName();
        }
        return new RowTypeInfo(types, names);
    }

    public static RowTypeInfo getRowTypeInfo(Schema schema) {
        TypeInformation<?>[] types = new TypeInformation<?>[schema.getColumns().size()];
        String[] names = new String[schema.getColumns().size()];

        for (int i = 0; i < schema.getColumns().size(); i++) {
            UnresolvedPhysicalColumn column = (UnresolvedPhysicalColumn) schema.getColumns().get(i);
            types[i] = ExternalTypeInfo.of(column.getDataType().getClass());
            names[i] = column.getName();
        }
        return new RowTypeInfo(types, names);
    }

    public static ResolvedSchema getOutputSchema(
            ResolvedSchema inputSchema, String[] resultCols, DataType[] resultTypes) {
        String[] reservedCols = inputSchema.getColumnNames().toArray(new String[0]);
        DataType[] reservedTypes = inputSchema.getColumnDataTypes().toArray(new DataType[0]);

        return ResolvedSchema.physical(
                ArrayUtils.addAll(reservedCols, resultCols),
                ArrayUtils.addAll(reservedTypes, resultTypes));
    }
}
