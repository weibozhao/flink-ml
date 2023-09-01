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

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.feature.lsh.MinHashLSHParams;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Calculates MinHash signatures for vectors.
 *
 * <p>Different from {@link org.apache.flink.ml.feature.lsh.MinHashLSH}, this algorithm works in a
 * transformer way, i.e., it directly outputs MinHash results.
 */
public class MinHash implements MinHashLSHParams<MinHash>, Transformer<MinHash> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public MinHash() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    public static MinHash load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Table in = inputs[0];
        final StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> data = tEnv.toDataStream(in);
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(in.getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.add(
                                inputTypeInfo.getFieldTypes(),
                                PrimitiveArrayTypeInfo.LONG_PRIMITIVE_ARRAY_TYPE_INFO),
                        ArrayUtils.add(inputTypeInfo.getFieldNames(), getOutputCol()));
        DataStream<Row> result =
                data.map(
                        new MinHashMapper(
                                getInputCol(),
                                getNumHashTables(),
                                getNumHashFunctionsPerTable(),
                                getSeed()),
                        outputTypeInfo);
        return new Table[] {tEnv.fromDataStream(result)};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    private static class MinHashMapper extends RichMapFunction<Row, Row> {
        private final String vectorCol;
        private final int numHashTables;
        private final int numProjectionsPerTable;
        private final long seed;
        private MinHashFunction minHashLSH;

        public MinHashMapper(
                String vectorCol, int numHashTables, int numProjectionsPerTable, long seed) {
            this.vectorCol = vectorCol;
            this.numHashTables = numHashTables;
            this.numProjectionsPerTable = numProjectionsPerTable;
            this.seed = seed;
        }

        @Override
        public void open(Configuration parameters) {
            minHashLSH = new MinHashFunction(seed, numProjectionsPerTable, numHashTables);
        }

        @Override
        public Row map(Row row) {
            long[] arr = minHashLSH.hashFunctionToLong(row.getFieldAs(vectorCol));
            //noinspection RedundantCast
            return Row.join(row, Row.of((Object) arr));
        }
    }
}
