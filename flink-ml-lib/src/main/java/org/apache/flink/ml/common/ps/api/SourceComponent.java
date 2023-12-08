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

package org.apache.flink.ml.common.ps.api;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.connector.source.Source;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.connector.file.src.reader.TextLineInputFormat;
import org.apache.flink.core.fs.Path;
import org.apache.flink.ml.common.ps.iterations.BaseComponent;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.types.Row;

/** Common stage. */
public class SourceComponent implements BaseComponent {

    protected String toName = "__flink_ml_source_name__";
    protected String filePath = null;
    protected String schemaStr = null;
    protected String delimiter = ",";

    public final SourceComponent output(String toName) {
        this.toName = toName;
        return this;
    }

    public final SourceComponent filePath(String filePath) {
        this.filePath = filePath;
        return this;
    }

    public final SourceComponent schema(String schema) {
        this.schemaStr = schema;
        return this;
    }

    public final SourceComponent delimiter(String delimiter) {
        this.delimiter = delimiter;
        return this;
    }

    public MLData source() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment(config);

        final String[] fieldInfo = schemaStr.split(",");
        String[] typeStrs = new String[fieldInfo.length];
        String[] names = new String[fieldInfo.length];
        TypeInformation<?>[] types = new TypeInformation<?>[fieldInfo.length];

        for (int i = 0; i < fieldInfo.length; ++i) {
            String[] tmp = fieldInfo[i].trim().split(" ");
            names[i] = tmp[0].trim();
            typeStrs[i] = tmp[1].trim();
            types[i] = parseType(typeStrs[i]);
        }
        Source<String, ?, ?> source =
                FileSource.forRecordStreamFormat(new TextLineInputFormat(), new Path(filePath))
                        .build();
        DataStream<Row> ds =
                env.fromSource(source, WatermarkStrategy.noWatermarks(), toName)
                        .map(
                                (MapFunction<String, Row>)
                                        s -> {
                                            String[] contents = s.split(delimiter);
                                            Row ret = new Row(contents.length);
                                            for (int i = 0; i < contents.length; ++i) {
                                                ret.setField(i, parse(typeStrs[i], contents[i]));
                                            }
                                            return ret;
                                        })
                        .returns(new RowTypeInfo(types, names));

        return new MLData(new DataStream[] {ds}, new String[] {toName});
    }

    public static Object parse(String type, String content) {
        switch (type.toUpperCase()) {
            case "STRING":
            case "VARCHAR":
                return content;
            case "INT":
            case "INTEGER":
                return Integer.parseInt(content);
            case "LONG":
            case "BIGINT":
                return Long.parseLong(content);
            case "BOOLEAN":
            case "BOOL":
                return Boolean.parseBoolean(content);
            case "TINYINT":
            case "BYTE":
                return Byte.parseByte(content);
            case "SMALLINT":
            case "SHORT":
                return Short.parseShort(content);
            case "FLOAT":
                return Float.parseFloat(content);
            case "DOUBLE":
                return Double.parseDouble(content);
            default:
                throw new RuntimeException("Not supported data type: " + type);
        }
    }

    public static TypeInformation<?> parseType(String type) {
        switch (type.toUpperCase()) {
            case "STRING":
            case "VARCHAR":
                return Types.STRING;
            case "INT":
            case "INTEGER":
                return Types.INT;
            case "LONG":
            case "BIGINT":
                return Types.LONG;
            case "BOOLEAN":
            case "BOOL":
                return Types.BOOLEAN;
            case "TINYINT":
            case "BYTE":
                return Types.BYTE;
            case "SMALLINT":
            case "SHORT":
                return Types.SHORT;
            case "DOUBLE":
                return Types.DOUBLE;
            case "FLOAT":
                return Types.FLOAT;
            default:
                throw new RuntimeException("Not supported data type: " + type);
        }
    }
}
