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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichCoGroupFunction;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.HeartbeatManagerOptions;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationConfig.OperatorLifeCycle;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.common.iteration.TerminateOnMaxIter;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;
import org.apache.flink.util.NumberSequenceIterator;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.List;
import java.util.Random;

/** Tests the {@link DataStreamUtils}. */
public class CoGroupWithIterationTest {
    private StreamExecutionEnvironment env;

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(HeartbeatManagerOptions.HEARTBEAT_TIMEOUT, 5000000L);
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.getConfig().enableObjectReuse();
        env.setRestartStrategy(RestartStrategies.noRestart());
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
    }

    private static class TrainIterationBody implements IterationBody {

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {

            DataStreamList feedbackVariableStream =
                    IterationBody.forEachRound(
                            dataStreams,
                            input -> {
                                DataStream<Tuple2<Long, DenseVector>> dataStream1 =
                                        variableStreams.get(0);
                                DataStream<Tuple2<Long, DenseVector>> dataStream2 =
                                        dataStreams.get(0);

                                DataStream<Tuple2<Long, DenseVector>> coResult =
                                        // dataStream1.coGroup(dataStream2)
                                        //	.where(
                                        //		(KeySelector <Tuple2 <Long, DenseVector>, Long>) t2 ->
                                        // t2.f0)
                                        //	.equalTo(
                                        //		(KeySelector <Tuple2 <Long, DenseVector>, Long>) t2 ->
                                        // t2.f0)
                                        //	.window(EndOfStreamWindows.get())
                                        //	.apply(new CoFunc1());

                                        DataStreamUtils.coGroup(
                                                dataStream1,
                                                dataStream2,
                                                (KeySelector<Tuple2<Long, DenseVector>, Long>)
                                                        t2 -> t2.f0,
                                                (KeySelector<Tuple2<Long, DenseVector>, Long>)
                                                        t2 -> t2.f0,
                                                new TupleTypeInfo<>(
                                                        Types.LONG,
                                                        TypeInformation.of(DenseVector.class)),
                                                new CoFunc1());
                                return DataStreamList.of(coResult);
                            });

            DataStream<Integer> terminationCriteria =
                    feedbackVariableStream
                            .get(0)
                            .flatMap(new TerminateOnMaxIter(3))
                            .returns(Types.INT);

            return new IterationBodyResult(
                    feedbackVariableStream, variableStreams, terminationCriteria);
        }
    }

    private static class CoFunc1
            extends RichCoGroupFunction<
                    Tuple2<Long, DenseVector>,
                    Tuple2<Long, DenseVector>,
                    Tuple2<Long, DenseVector>> {
        @Override
        public void coGroup(
                Iterable<Tuple2<Long, DenseVector>> iterable,
                Iterable<Tuple2<Long, DenseVector>> iterable1,
                Collector<Tuple2<Long, DenseVector>> collector) {
            Long id = null;
            DenseVector denseVector = null;
            for (Tuple2<Long, DenseVector> iter : iterable) {
                id = iter.f0;
                if (denseVector == null) {
                    denseVector = iter.f1;
                } else {
                    BLAS.axpy(1.0, iter.f1, denseVector);
                }
            }
            for (Tuple2<Long, DenseVector> iter : iterable1) {
                id = iter.f0;
                if (denseVector == null) {
                    denseVector = iter.f1;
                } else {
                    BLAS.axpy(1.0, iter.f1, denseVector);
                }
            }
            collector.collect(Tuple2.of(id, denseVector));
        }
    }

    @Test
    public void test100() throws Exception {
        for (int i = 0; i < 100; ++i) {
            testCoGroupWithIteration();
        }
    }

    @Test
    public void testCoGroupWithIteration() throws Exception {
        DataStream<Tuple2<Long, DenseVector>> dataStream1 =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 5L), Types.LONG)
                        .map(
                                new MapFunction<Long, Tuple2<Long, DenseVector>>() {
                                    final Random rand = new Random();

                                    @Override
                                    public Tuple2<Long, DenseVector> map(Long aLong) {
                                        return Tuple2.of(
                                                aLong,
                                                new DenseVector(
                                                        new double[] {
                                                            rand.nextDouble(), rand.nextDouble()
                                                        }));
                                    }
                                });
        DataStream<Tuple2<Long, DenseVector>> dataStream2 =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 5L), Types.LONG)
                        .map(
                                new MapFunction<Long, Tuple2<Long, DenseVector>>() {
                                    final Random rand = new Random();

                                    @Override
                                    public Tuple2<Long, DenseVector> map(Long aLong) {
                                        return Tuple2.of(
                                                aLong,
                                                new DenseVector(
                                                        new double[] {
                                                            rand.nextDouble(), rand.nextDouble()
                                                        }));
                                    }
                                });
        DataStreamList coResult =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(dataStream1),
                        ReplayableDataStreamList.notReplay(dataStream2),
                        IterationConfig.newBuilder()
                                .setOperatorLifeCycle(OperatorLifeCycle.PER_ROUND)
                                .build(),
                        new TrainIterationBody());

        List<Tuple2<Long, DenseVector>> list =
                IteratorUtils.toList(coResult.get(0).executeAndCollect());

        long sum = 1;
        for (Tuple2<Long, DenseVector> t2 : list) {
            sum += t2.f0;
        }
        System.out.println(sum);
    }
}
