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

package org.apache.flink.ml.recommendation;

import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.HeartbeatManagerOptions;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationConfig.OperatorLifeCycle;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.iteration.TerminateOnMaxIter;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;
import org.apache.flink.util.NumberSequenceIterator;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;

/** Tests the {@link DataStreamUtils}. */
public class MapPatitionWithIterationTest {
    private StreamExecutionEnvironment env;

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(HeartbeatManagerOptions.HEARTBEAT_TIMEOUT, 5000000L);
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.getConfig().enableObjectReuse();
        env.setRestartStrategy(RestartStrategies.noRestart());
        env.setParallelism(1);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
    }

    @Test
    public void testIterationWithMapPartition() throws Exception {
        DataStream<Long> broadcast =
            env.fromParallelCollection(new NumberSequenceIterator(0L, 5L), Types.LONG);
        DataStream<Long> input =
            env.fromParallelCollection(new NumberSequenceIterator(0L, 5L), Types.LONG);
        DataStreamList coResult =
            Iterations.iterateBoundedStreamsUntilTermination(
                DataStreamList.of(input),
                ReplayableDataStreamList.notReplay(broadcast),
                IterationConfig.newBuilder()
                    .setOperatorLifeCycle(OperatorLifeCycle.PER_ROUND)
                    .build(),
                new IterationBodyWithMapPartition());

        List<Integer> counts = IteratorUtils.toList(coResult.get(0).executeAndCollect());
        System.out.println(counts.size());
    }

    private static class IterationBodyWithMapPartition implements IterationBody {

        @Override
        public IterationBodyResult process(
            DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<Long> dataStream1 = variableStreams.get(0);

            DataStream<Long> coResult =
                DataStreamUtils.mapPartition(
                    dataStream1,
                    new MapPartitionFunction <Long, Long>() {
                        @Override
                        public void mapPartition(Iterable <Long> iterable, Collector <Long> collector)
                            throws Exception {
                            for (Long iter: iterable) {
                                collector.collect(iter);
                            }
                        }
                    });

            DataStream<Integer> terminationCriteria =
                coResult.<Long>flatMap(new TerminateOnMaxIter(2)).returns(Types.INT);

            return new IterationBodyResult(
                DataStreamList.of(coResult), variableStreams, terminationCriteria);
        }
    }
}
