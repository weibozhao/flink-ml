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

package org.apache.flink.ml.benchmark.datagenerator.common;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.benchmark.datagenerator.InputDataGenerator;
import org.apache.flink.ml.benchmark.datagenerator.param.HasVectorDim;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.NumberSequenceIterator;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * A DataGenerator which creates a table of features, label and weight.
 */
public class ColsWithLabelGenerator
        implements InputDataGenerator<ColsWithLabelGenerator>,
                HasVectorDim<ColsWithLabelGenerator> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public ColsWithLabelGenerator() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] getData(StreamTableEnvironment tEnv) {

        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);
        TypeInformation<?>[] types;
        String[] colNames;
        if (getColNames().length == 2) {
            colNames = new String[getColNames()[0].length + 1];

            types = new TypeInformation<?>[getColNames()[0].length + 1];
            for (int i = 0; i < getColNames()[0].length; ++i) {
                colNames[i] = getColNames()[0][i];
                types[i] = Types.DOUBLE;
            }
            colNames[colNames.length - 1] = getColNames()[1][0];
            types[colNames.length - 1] = Types.DOUBLE;
        } else {
            colNames = getColNames()[0];
            types = new TypeInformation<?>[getColNames()[0].length];
            for (int i = 0; i < colNames.length; ++i) {
                types[i] = Types.DOUBLE;
            }
        }
        DataStream<Row> dataStream =
                env.fromParallelCollection(
                                new NumberSequenceIterator(1L, getNumValues()),
                                BasicTypeInfo.LONG_TYPE_INFO)
                        .map(
                                new ColsGenerator(getSeed(), colNames),
                                new RowTypeInfo(types, colNames));

        Table dataTable = tEnv.fromDataStream(dataStream);

        return new Table[] {dataTable};
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    private static class ColsGenerator extends RichMapFunction<Long, Row> {
        private final long initSeed;
        private final String[] colNames;
        private Random random;

        private ColsGenerator(long initSeed, String[] colNames) {
            this.initSeed = initSeed;
            this.colNames = colNames;
        }

        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
            int index = getRuntimeContext().getIndexOfThisSubtask();
            random = new Random(Tuple2.of(initSeed, index).hashCode());
        }

        @Override
        public Row map(Long ignored) {
            Row ret = new Row(colNames.length);
            for (int i = 0; i < colNames.length - 1; ++i) {
                ret.setField(i, random.nextDouble());
            }
            ret.setField(colNames.length - 1, random.nextDouble() > 0.5 ? 1.0 : 0.0);
            return ret;
        }
    }
}
