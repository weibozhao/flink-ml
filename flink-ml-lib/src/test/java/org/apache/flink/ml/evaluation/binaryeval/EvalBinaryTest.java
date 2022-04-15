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

package org.apache.flink.ml.evaluation.binaryeval;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.common.datastream.EndOfStreamWindows;
import org.apache.flink.ml.feature.minmaxscaler.MinMaxScaler;
import org.apache.flink.ml.feature.minmaxscaler.MinMaxScalerModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.ml.util.StageTestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.functions.windowing.WindowFunction;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collector;

import static org.junit.Assert.assertEquals;

/** Tests {@link EvalBinaryClass}. */
public class EvalBinaryTest {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table trainDataTable;
    private static final List<Row> TRAIN_DATA =
            new ArrayList<>(
                    Arrays.asList(
        				Row.of(1.0, Vectors.dense(0.1, 0.9)),
                        Row.of(1.0, Vectors.dense(0.8, 0.2)),
                        Row.of(1.0, Vectors.dense(0.7, 0.3)),
                        Row.of(0.0, Vectors.dense(0.75, 0.25)),
                        Row.of(0.0, Vectors.dense(0.6, 0.4)),
                        Row.of(1.0, Vectors.dense(0.65, 0.35)),
                        Row.of(1.0, Vectors.dense(0.55, 0.45)),
                        Row.of(0.0, Vectors.dense(0.4, 0.6)),
                        Row.of(0.0, Vectors.dense(0.3, 0.7)),
                        Row.of(1.0, Vectors.dense(0.35, 0.65)),
                        Row.of(0.0, Vectors.dense(0.2, 0.8)),
                        Row.of(1.0, Vectors.dense(0.1, 0.9))));

    private static final List<Row> TRAIN_DATA_1 =
        new ArrayList<>(
            Arrays.asList(
                Row.of(1.0, Vectors.dense(0.1, 0.9)),
                Row.of(1.0, Vectors.dense(0.1, 0.9)),
                Row.of(1.0, Vectors.dense(0.1, 0.9)),
                Row.of(0.0, Vectors.dense(0.25, 0.75)),
                Row.of(0.0, Vectors.dense(0.4, 0.6)),
                Row.of(1.0, Vectors.dense(0.1, 0.9)),
                Row.of(1.0, Vectors.dense(0.1, 0.9)),
                Row.of(0.0, Vectors.dense(0.6, 0.4)),
                Row.of(0.0, Vectors.dense(0.7, 0.3)),
                Row.of(1.0, Vectors.dense(0.1, 0.9)),
                Row.of(0.0, Vectors.dense(0.8, 0.2)),
                Row.of(1.0, Vectors.dense(0.1, 0.9))));
    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);
        trainDataTable = tEnv.fromDataStream(env.fromCollection(TRAIN_DATA_1)).as("label", "raw");
    }

    @Test
    public void testEval() throws Exception {
        EvalBinaryClass eval = new EvalBinaryClass().setLabelCol("label").setRawPredictionCol("raw");
        eval.transform(trainDataTable);
        env.execute();
    }

    @Test
    public void test() throws Exception {
        DataStream <Tuple2 <Long, Long>> input = env.fromElements(1L, 2L, 3L, 4L).map(x -> Tuple2.of(x / 2, x)).returns(new TupleTypeInfo <>(
            BasicTypeInfo.LONG_TYPE_INFO,
            BasicTypeInfo.LONG_TYPE_INFO
        ));

        DataStream <Object> output = input.keyBy(new KeySelector <Tuple2 <Long, Long>, Long>() {
                @Override
                public Long getKey(Tuple2 <Long, Long> value) throws Exception {
                    return value.f0;
                }
            })
            .window(EndOfStreamWindows.get())
            .apply(new WindowFunction <Tuple2 <Long, Long>, Object, Long, TimeWindow>() {
                @Override
                public void apply(Long aLong, TimeWindow window, Iterable <Tuple2 <Long, Long>> input,
                                  org.apache.flink.util.Collector <Object> out) throws Exception {
                    for (Tuple2 <Long, Long> val : input) {
                        System.out.println(aLong + " ----- " + val);
                    }
                }
            })
            .returns(Object.class);
        output.addSink(new SinkFunction <Object>() {});
        env.execute();

    }
}
