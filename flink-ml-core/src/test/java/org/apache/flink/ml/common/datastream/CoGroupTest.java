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

import org.apache.flink.api.common.functions.RichCoGroupFunction;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.HeartbeatManagerOptions;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.common.iteration.TerminateOnMaxIter;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/** Tests the {@link DataStreamUtils}. */
public class CoGroupTest {
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table trainDataTable;

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(HeartbeatManagerOptions.HEARTBEAT_TIMEOUT, 5000000L);
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.getConfig().enableObjectReuse();
        env.setRestartStrategy(RestartStrategies.noRestart());
        env.setParallelism(1);
        // env.setRuntimeMode(RuntimeExecutionMode.BATCH);
        env.getCheckpointConfig().disableCheckpointing();
        env.setRestartStrategy(RestartStrategies.noRestart());

        Random rand = new Random(0);
        List<Row> trainData = new ArrayList<>();
        for (int i = 0; i < 2000000; ++i) {
            trainData.add(
                    Row.of((long) rand.nextInt(200), (long) rand.nextInt(200), rand.nextDouble()));
        }
        Collections.shuffle(trainData);
        tEnv = StreamTableEnvironment.create(env);

        trainDataTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                trainData,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            Types.LONG, Types.LONG, Types.DOUBLE
                                        },
                                        new String[] {"user_id", "item_id", "rating"})));
    }

    @Test
    public void testCoGroup() throws Exception {
        DataStream<Row> data1 = tEnv.toDataStream(trainDataTable);
        DataStream<Row> data2 = tEnv.toDataStream(trainDataTable);
        DataStream<?> result =
                data1.coGroup(data2)
                        .where(
                                new KeySelector<Row, Long>() {
                                    @Override
                                    public Long getKey(Row row) throws Exception {
                                        return row.getFieldAs(0);
                                    }
                                })
                        .equalTo(
                                new KeySelector<Row, Long>() {
                                    @Override
                                    public Long getKey(Row row) throws Exception {
                                        return row.getFieldAs(1);
                                    }
                                })
                        .window(EndOfStreamWindows.get())
                        .apply(
                                new RichCoGroupFunction<Row, Row, Integer>() {
                                    private long time;

                                    @Override
                                    public void open(Configuration parameters) throws Exception {
                                        super.open(parameters);
                                        time = System.currentTimeMillis();
                                    }

                                    @Override
                                    public void close() throws Exception {
                                        super.close();
                                        System.out.println(
                                                "yty : "
                                                        + (System.currentTimeMillis() - time)
                                                                / 1000);
                                    }

                                    @Override
                                    public void coGroup(
                                            Iterable<Row> iterable,
                                            Iterable<Row> iterable1,
                                            Collector<Integer> collector) {}
                                });
        IteratorUtils.toList(result.executeAndCollect());
    }

    @Test
    public void testCoGroupWithIteration() throws Exception {
        DataStream<Row> data1 = tEnv.toDataStream(trainDataTable);
        DataStream<Row> data2 = tEnv.toDataStream(trainDataTable);
        DataStreamList coResult =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(data1),
                        ReplayableDataStreamList.notReplay(data2),
                        IterationConfig.newBuilder().build(),
                        new TrainIterationBody());

        List<Integer> counts = IteratorUtils.toList(coResult.get(0).executeAndCollect());
        System.out.println(counts.size());
    }

    private static class TrainIterationBody implements IterationBody {

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {

            DataStreamList feedbackVariableStream =
                    IterationBody.forEachRound(
                            dataStreams,
                            input -> {
                                DataStream<Row> dataStream1 = variableStreams.get(0);
                                DataStream<Row> dataStream2 = dataStreams.get(0);

                                DataStream<Row> coResult =
                                        dataStream1
                                                .coGroup(dataStream2)
                                                .where(
                                                        (KeySelector<Row, Long>)
                                                                t2 -> t2.getFieldAs(0))
                                                .equalTo(
                                                        (KeySelector<Row, Long>)
                                                                t2 -> t2.getFieldAs(1))
                                                .window(EndOfStreamWindows.get())
                                                .apply(
                                                        new RichCoGroupFunction<Row, Row, Row>() {
                                                            @Override
                                                            public void coGroup(
                                                                    Iterable<Row> iterable,
                                                                    Iterable<Row> iterable1,
                                                                    Collector<Row> collector) {
                                                                for (Row row : iterable1) {
                                                                    collector.collect(row);
                                                                }
                                                            }
                                                        });
                                return DataStreamList.of(coResult);
                            });

            DataStream<Integer> terminationCriteria =
                    feedbackVariableStream
                            .get(0)
                            .flatMap(new TerminateOnMaxIter(2))
                            .returns(Types.INT);

            return new IterationBodyResult(
                    feedbackVariableStream, feedbackVariableStream, terminationCriteria);
        }
    }
}
