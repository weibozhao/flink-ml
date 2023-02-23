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

package org.apache.flink.test.iteration;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationConfig.OperatorLifeCycle;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.iteration.TerminateOnMaxIter;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.testutils.junit.SharedObjects;
import org.apache.flink.testutils.junit.SharedReference;
import org.apache.flink.util.TestLogger;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import static org.apache.flink.test.iteration.UnboundedStreamIterationITCase.createMiniClusterConfiguration;
import static org.junit.Assert.assertEquals;

/** Tests using {@link BroadcastUtils#withBroadcastStream} in iterations. */
public class BoundedIterationWithBroadcastITCase extends TestLogger {

    @Rule public final SharedObjects sharedObjects = SharedObjects.create();

    private MiniCluster miniCluster;

    private SharedReference<BlockingQueue<Long>> result;

    @Before
    public void setup() throws Exception {
        miniCluster = new MiniCluster(createMiniClusterConfiguration(1, 1));
        miniCluster.start();

        result = sharedObjects.add(new LinkedBlockingQueue<>());
    }

    @After
    public void teardown() throws Exception {
        if (miniCluster != null) {
            miniCluster.close();
        }
    }

    @Test
    public void testIterationWithBroadcastJobGraph() throws Exception {
        JobGraph jobGraph = createIterationWithBroadcastJobGraph();
        miniCluster.executeJobBlocking(jobGraph);

        List<Long> collectResult = new ArrayList<>(3);
        result.get().drainTo(collectResult);
        assertEquals(3, collectResult.size());
        for (long value : collectResult) {
            assertEquals(1L, value);
        }
    }

    private JobGraph createIterationWithBroadcastJobGraph() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        DataStream<Long> broadcast = env.fromElements(1L);
        DataStream<Long> input = env.fromElements(1L);
        DataStreamList coResult =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(input),
                        ReplayableDataStreamList.replay(broadcast),
                        IterationConfig.newBuilder()
                                .setOperatorLifeCycle(OperatorLifeCycle.PER_ROUND)
                                .build(),
                        new IterationBodyWithBroadcast());

        coResult.<Long>get(0).addSink(new LongSink(result));
        return env.getStreamGraph().getJobGraph();
    }

    private static class IterationBodyWithBroadcast implements IterationBody {

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<Long> variableStream = variableStreams.get(0);
            DataStream<Long> constantStream = dataStreams.get(0);

            DataStream<Long> feedback =
                    BroadcastUtils.withBroadcastStream(
                                    Arrays.asList(variableStream),
                                    Collections.singletonMap("broadcast", constantStream),
                                    inputList -> {
                                        DataStream<Long> stream =
                                                (DataStream<Long>) inputList.get(0);

                                        return stream.map((MapFunction<Long, Long>) x -> x);
                                    })
                            .map(x -> x)
                            .returns(Types.LONG);

            DataStream<Integer> terminationCriteria =
                    feedback.<Long>flatMap(new TerminateOnMaxIter(2)).returns(Types.INT);

            return new IterationBodyResult(
                    DataStreamList.of(feedback), DataStreamList.of(feedback), terminationCriteria);
        }
    }

    private static class LongSink implements SinkFunction<Long> {
        private final SharedReference<BlockingQueue<Long>> collectedLong;

        public LongSink(SharedReference<BlockingQueue<Long>> collectedLong) {
            this.collectedLong = collectedLong;
        }

        @Override
        public void invoke(Long value, Context context) {
            collectedLong.get().add(value);
        }
    }
}
