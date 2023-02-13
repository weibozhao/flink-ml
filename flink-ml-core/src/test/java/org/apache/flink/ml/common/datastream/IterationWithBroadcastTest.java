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

import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationConfig.OperatorLifeCycle;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;
import org.apache.flink.util.NumberSequenceIterator;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/** Tests the {@link DataStreamUtils}. */
public class IterationWithBroadcastTest {
    private StreamExecutionEnvironment env;

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.getConfig().enableObjectReuse();
        env.setParallelism(1);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
    }

    @Test
    public void testIterationWithBroadcast() throws Exception {
        DataStream <Long> broadcast =
            env.fromParallelCollection(new NumberSequenceIterator(0L, 2L), Types.LONG);
        DataStream <Long> dataStream1 =
            env.fromParallelCollection(new NumberSequenceIterator(0L, 5L), Types.LONG);
        DataStreamList coResult =
            Iterations.iterateBoundedStreamsUntilTermination(
                DataStreamList.of(dataStream1),
                ReplayableDataStreamList.replay(broadcast),
                IterationConfig.newBuilder()
                    .setOperatorLifeCycle(OperatorLifeCycle.PER_ROUND)
                    .build(),
                new IterationBodyWithBroadcast());

        List <Integer> counts = IteratorUtils.toList(coResult.get(0).executeAndCollect());
        System.out.println(counts.size());
    }

    private static class IterationBodyWithBroadcast implements IterationBody {

        @Override
        public IterationBodyResult process(
            DataStreamList variableStreams, DataStreamList dataStreams) {

            DataStreamList feedbackVariableStream =
                IterationBody.forEachRound(
                    dataStreams,
                    input -> {
                        DataStream <Long> dataStream1 = variableStreams.get(0);

                        DataStream <Long> coResult =
                            BroadcastUtils.withBroadcastStream(
                                Arrays.asList(dataStream1),
                                Collections.singletonMap(
                                    "broadcast", dataStreams.get(0)),
                                inputList -> {
                                    DataStream <Long> data1 =
                                        (DataStream <Long>) inputList.get(0);

                                    return data1.map(
                                        new RichMapFunction <Long, Long>() {
                                            @Override
                                            public Long map(
                                                Long longDenseVectorTuple2)
                                                throws Exception {
                                                System.out.println("broadcast var : " +
                                                    getRuntimeContext().getBroadcastVariable("broadcast"));
                                                return longDenseVectorTuple2;
                                            }
                                        });
                                });

                        return DataStreamList.of(
                            coResult.filter(
                                (FilterFunction <Long>)
                                    longDenseVectorTuple2 ->
                                        longDenseVectorTuple2 > 0L));
                    });

            DataStream <Integer> terminationCriteria =
                feedbackVariableStream
                    .get(0)
                    .flatMap(new TerminateOnMaxIter(5))
                    .returns(Types.INT);

            return new IterationBodyResult(
                feedbackVariableStream, variableStreams, terminationCriteria);
        }
    }

    public static class TerminateOnMaxIter
        implements IterationListener <Integer>, FlatMapFunction <Object, Integer> {

        private final int maxIter;

        public TerminateOnMaxIter(Integer maxIter) {
            this.maxIter = maxIter;
        }

        @Override
        public void flatMap(Object value, Collector <Integer> out) {}

        @Override
        public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector <Integer> collector) {
            System.out.println(epochWatermark);
            if ((epochWatermark + 1) < maxIter) {
                collector.collect(0);
            }
        }

        @Override
        public void onIterationTerminated(Context context, Collector <Integer> collector) {}
    }
}
