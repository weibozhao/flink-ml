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
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichCoGroupFunction;
import org.apache.flink.api.common.functions.RichReduceFunction;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.HeartbeatManagerOptions;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationConfig.OperatorLifeCycle;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.iteration.TerminateOnMaxIter;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.transformations.PhysicalTransformation;
import org.apache.flink.util.Collector;
import org.apache.flink.util.NumberSequenceIterator;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/** Tests the {@link DataStreamUtils}. */
public class CoGroupTest {
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
    public void testCoGroupWithBroadcast() throws Exception {
        DataStream<Long> broadcast =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 2L), Types.LONG)
                        .map(
                                new MapFunction<Long, Long>() {
                                    @Override
                                    public Long map(Long aLong) throws Exception {
                                        Thread.sleep(10);
                                        return aLong;
                                    }
                                });
        DataStream<Long> dataStream1 =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 5L), Types.LONG);
        DataStream<Long> dataStream2 =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 5L), Types.LONG);

        DataStream<Integer> coResult =
                BroadcastUtils.withBroadcastStream(
                        Arrays.asList(dataStream1, dataStream2),
                        Collections.singletonMap("broadcast", dataStream1),
                        inputList -> {
                            DataStream<Long> data1 = (DataStream<Long>) inputList.get(0);
                            DataStream<Long> data2 = (DataStream<Long>) inputList.get(1);

                            return data1.coGroup(data2)
                                    .where((KeySelector<Long, Long>) aLong -> aLong)
                                    .equalTo((KeySelector<Long, Long>) aLong -> aLong)
                                    .window(EndOfStreamWindows.get())
                                    .apply(
                                            new RichCoGroupFunction<Long, Long, Integer>() {
                                                @Override
                                                public void coGroup(
                                                        Iterable<Long> iterable,
                                                        Iterable<Long> iterable1,
                                                        Collector<Integer> collector) {
                                                    collector.collect(
                                                            Integer.valueOf(
                                                                    getRuntimeContext()
                                                                            .getBroadcastVariable(
                                                                                    "broadcast")
                                                                            .get(1)
                                                                            .toString()));
                                                }
                                            });
                        });

        List<Integer> counts = IteratorUtils.toList(coResult.executeAndCollect());
        assertEquals(6, counts.size());
        for (int count : counts) {
            assertEquals(1, count);
        }
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
                                        variableStreams.get(1);

                                DataStream<Tuple2<Long, DenseVector>> coResult =
                                        dataStream1
                                                .coGroup(dataStream2)
                                                .where(
                                                        (KeySelector<
                                                                        Tuple2<Long, DenseVector>,
                                                                        Long>)
                                                                t2 -> t2.f0)
                                                .equalTo(
                                                        (KeySelector<
                                                                        Tuple2<Long, DenseVector>,
                                                                        Long>)
                                                                t2 -> t2.f0)
                                                .window(EndOfStreamWindows.get())
                                                .apply(
                                                        new RichCoGroupFunction<
                                                                Tuple2<Long, DenseVector>,
                                                                Tuple2<Long, DenseVector>,
                                                                Tuple2<Long, DenseVector>>() {
                                                            @Override
                                                            public void coGroup(
                                                                    Iterable<
                                                                                    Tuple2<
                                                                                            Long,
                                                                                            DenseVector>>
                                                                            iterable,
                                                                    Iterable<
                                                                                    Tuple2<
                                                                                            Long,
                                                                                            DenseVector>>
                                                                            iterable1,
                                                                    Collector<
                                                                                    Tuple2<
                                                                                            Long,
                                                                                            DenseVector>>
                                                                            collector) {
                                                                for (Tuple2<Long, DenseVector>
                                                                        iter : iterable) {
                                                                    if (iter == null) {
                                                                        continue;
                                                                    }
                                                                    collector.collect(iter);
                                                                    System.out.println(
                                                                            getRuntimeContext()
                                                                                            .getIndexOfThisSubtask()
                                                                                    + " "
                                                                                    + iter);
                                                                }
                                                                for (Tuple2<Long, DenseVector>
                                                                        iter : iterable1) {
                                                                    if (iter == null) {
                                                                        continue;
                                                                    }
                                                                    System.out.println(
                                                                            getRuntimeContext()
                                                                                            .getIndexOfThisSubtask()
                                                                                    + " "
                                                                                    + iter);
                                                                    collector.collect(iter);
                                                                }
                                                            }
                                                        });
                                return DataStreamList.of(
                                        coResult.filter(
                                                (FilterFunction<Tuple2<Long, DenseVector>>)
                                                        longDenseVectorTuple2 ->
                                                                longDenseVectorTuple2.f0 > 0L),
                                        coResult.filter(
                                                (FilterFunction<Tuple2<Long, DenseVector>>)
                                                        longDenseVectorTuple2 ->
                                                                longDenseVectorTuple2.f0 < 0L));
                            });

            DataStream<Integer> terminationCriteria =
                    feedbackVariableStream
                            .get(0)
                            .flatMap(new TerminateOnMaxIter2(2))
                            .returns(Types.INT);

            return new IterationBodyResult(
                    feedbackVariableStream, variableStreams, terminationCriteria);
        }
    }

    @Test
    public void testCoGroupWithIteration() throws Exception {
        DataStream<Long> broadcast =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 2L), Types.LONG);
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
                                                -aLong,
                                                new DenseVector(
                                                        new double[] {
                                                            rand.nextDouble(), rand.nextDouble()
                                                        }));
                                    }
                                });
        DataStreamList coResult =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(dataStream1, dataStream2),
                        ReplayableDataStreamList.notReplay(broadcast),
                        IterationConfig.newBuilder().build(),
                        new TrainIterationBody());

        List<Integer> counts = IteratorUtils.toList(coResult.get(0).executeAndCollect());
        System.out.println(counts.size());
    }

    private static class TrainIterationBodyWithBroadcast implements IterationBody {

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {

            DataStreamList feedbackVariableStream =
                    IterationBody.forEachRound(
                            DataStreamList.of(
                                    variableStreams.get(0),
                                    variableStreams.get(1),
                                    dataStreams.get(0)),
                            input -> {
                                DataStream<Tuple2<Long, DenseVector>> dataStream1 = input.get(0);
                                DataStream<Tuple2<Long, DenseVector>> dataStream2 = input.get(1);

                                DataStream<Tuple2<Long, DenseVector>> coResult =
                                        BroadcastUtils.withBroadcastStream(
                                                Arrays.asList(dataStream1, dataStream2),
                                                Collections.singletonMap("broadcast", input.get(0)),
                                                inputList -> {
                                                    DataStream<Tuple2<Long, DenseVector>> data1 =
                                                            (DataStream<Tuple2<Long, DenseVector>>)
                                                                    inputList.get(0);
                                                    DataStream<Tuple2<Long, DenseVector>> data2 =
                                                            (DataStream<Tuple2<Long, DenseVector>>)
                                                                    inputList.get(1);

                                                    return data1.coGroup(data2)
                                                            .where(
                                                                    (KeySelector<
                                                                                    Tuple2<
                                                                                            Long,
                                                                                            DenseVector>,
                                                                                    Long>)
                                                                            t2 -> t2.f0)
                                                            .equalTo(
                                                                    (KeySelector<
                                                                                    Tuple2<
                                                                                            Long,
                                                                                            DenseVector>,
                                                                                    Long>)
                                                                            t2 -> t2.f0)
                                                            .window(EndOfStreamWindows.get())
                                                            .apply(
                                                                    new RichCoGroupFunction<
                                                                            Tuple2<
                                                                                    Long,
                                                                                    DenseVector>,
                                                                            Tuple2<
                                                                                    Long,
                                                                                    DenseVector>,
                                                                            Tuple2<
                                                                                    Long,
                                                                                    DenseVector>>() {
                                                                        @Override
                                                                        public void coGroup(
                                                                                Iterable<
                                                                                                Tuple2<
                                                                                                        Long,
                                                                                                        DenseVector>>
                                                                                        iterable,
                                                                                Iterable<
                                                                                                Tuple2<
                                                                                                        Long,
                                                                                                        DenseVector>>
                                                                                        iterable1,
                                                                                Collector<
                                                                                                Tuple2<
                                                                                                        Long,
                                                                                                        DenseVector>>
                                                                                        collector) {
                                                                            List<
                                                                                            Tuple2<
                                                                                                    Long,
                                                                                                    DenseVector>>
                                                                                    b =
                                                                                            getRuntimeContext()
                                                                                                    .getBroadcastVariable(
                                                                                                            "broadcast");
                                                                            System.out.println(
                                                                                    getRuntimeContext()
                                                                                                    .getIndexOfThisSubtask()
                                                                                            + " received broadcast variable: "
                                                                                            + b);
                                                                            for (Tuple2<
                                                                                            Long,
                                                                                            DenseVector>
                                                                                    iter :
                                                                                            iterable) {
                                                                                if (iter == null) {
                                                                                    continue;
                                                                                }
                                                                                collector.collect(
                                                                                        iter);
                                                                                System.out.println(
                                                                                        getRuntimeContext()
                                                                                                        .getIndexOfThisSubtask()
                                                                                                + " "
                                                                                                + iter);
                                                                            }
                                                                            for (Tuple2<
                                                                                            Long,
                                                                                            DenseVector>
                                                                                    iter :
                                                                                            iterable1) {
                                                                                if (iter == null) {
                                                                                    continue;
                                                                                }
                                                                                System.out.println(
                                                                                        getRuntimeContext()
                                                                                                        .getIndexOfThisSubtask()
                                                                                                + " "
                                                                                                + iter);
                                                                                collector.collect(
                                                                                        iter);
                                                                            }
                                                                        }
                                                                    });
                                                });
                                return DataStreamList.of(
                                        coResult.filter(
                                                (FilterFunction<Tuple2<Long, DenseVector>>)
                                                        longDenseVectorTuple2 ->
                                                                longDenseVectorTuple2.f0 > 0L),
                                        coResult.filter(
                                                (FilterFunction<Tuple2<Long, DenseVector>>)
                                                        longDenseVectorTuple2 ->
                                                                longDenseVectorTuple2.f0 < 0L));
                            });

            DataStream<Integer> terminationCriteria =
                    feedbackVariableStream
                            .get(0)
                            .flatMap(new TerminateOnMaxIter2(2))
                            .returns(Types.INT);

            return new IterationBodyResult(
                    feedbackVariableStream, variableStreams, terminationCriteria);
        }
    }

    private <T1, T2> void coGroupImpl(DataStream<T1> input1, DataStream<T2> input2) {}

    @Test
    public void testCoGroupWithIterationAndBroadcast() throws Exception {
        DataStream<Long> broadcast =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 1L), Types.LONG);
        DataStream<Tuple2<Long, DenseVector>> dataStream1 =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 1L), Types.LONG)
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
                env.fromParallelCollection(new NumberSequenceIterator(0L, 1L), Types.LONG)
                        .map(
                                new MapFunction<Long, Tuple2<Long, DenseVector>>() {
                                    final Random rand = new Random();

                                    @Override
                                    public Tuple2<Long, DenseVector> map(Long aLong) {
                                        return Tuple2.of(
                                                -aLong,
                                                new DenseVector(
                                                        new double[] {
                                                            rand.nextDouble(), rand.nextDouble()
                                                        }));
                                    }
                                });
        DataStreamList coResult =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(dataStream1, dataStream2),
                        ReplayableDataStreamList.replay(broadcast),
                        IterationConfig.newBuilder()
                                .setOperatorLifeCycle(OperatorLifeCycle.PER_ROUND)
                                .build(),
                        new TrainIterationBodyWithBroadcast());

        // coResult.get(0).addSink(new SinkFunction <Object>() {});
        // System.out.println(env.getExecutionPlan());

        List<Integer> counts = IteratorUtils.toList(coResult.get(0).executeAndCollect());
        System.out.println(counts.size());
    }

    private static class TrainIterationBodyWithBroadcastLong implements IterationBody {

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {

            DataStreamList feedbackVariableStream =
                    IterationBody.forEachRound(
                            DataStreamList.of(
                                    variableStreams.get(0),
                                    variableStreams.get(1),
                                    dataStreams.get(0)),
                            input -> {
                                DataStream<Long> dataStream1 = input.get(0);
                                DataStream<Long> dataStream2 = input.get(1);
                                DataStream<Long> dataStream3 = input.get(2);
                                DataStream<Long> broad =
                                        dataStream2
                                                .coGroup(dataStream3)
                                                .where((KeySelector<Long, Long>) t2 -> t2)
                                                .equalTo((KeySelector<Long, Long>) t2 -> t2)
                                                .window(EndOfStreamWindows.get())
                                                .apply(
                                                        new RichCoGroupFunction<
                                                                Long, Long, Long>() {
                                                            @Override
                                                            public void coGroup(
                                                                    Iterable<Long> iterable,
                                                                    Iterable<Long> iterable1,
                                                                    Collector<Long> collector) {
                                                                System.out.println(
                                                                        getRuntimeContext()
                                                                                .getClass()
                                                                                .getSimpleName());
                                                                for (Long iter : iterable) {
                                                                    if (iter == null) {
                                                                        continue;
                                                                    }
                                                                    collector.collect(iter);
                                                                    System.out.println(
                                                                            "task id: "
                                                                                    + getRuntimeContext()
                                                                                            .getIndexOfThisSubtask()
                                                                                    + " value: "
                                                                                    + iter);
                                                                }
                                                                for (Long iter : iterable1) {
                                                                    if (iter == null) {
                                                                        continue;
                                                                    }
                                                                    System.out.println(
                                                                            "task id: "
                                                                                    + getRuntimeContext()
                                                                                            .getIndexOfThisSubtask()
                                                                                    + " value: "
                                                                                    + iter);
                                                                    collector.collect(iter);
                                                                }
                                                            }
                                                        });
                                // 0-epoch: the result should be: 0 0 1 1 2 2 3 3 4 4 5 5
                                // 1-epoch: one of  the input should be 0 1 2 4 5, the other one
                                // should be emtpy
                                DataStream<Long> coResult =
                                        BroadcastUtils.withBroadcastStream(
                                                Arrays.asList(dataStream1, dataStream2),
                                                Collections.singletonMap("broadcast", broad),
                                                inputList -> {
                                                    DataStream<Long> data1 =
                                                            (DataStream<Long>) inputList.get(0);
                                                    DataStream<Long> data2 =
                                                            (DataStream<Long>) inputList.get(1);

                                                    return data1.coGroup(data2)
                                                            .where(
                                                                    (KeySelector<Long, Long>)
                                                                            t2 -> t2)
                                                            .equalTo(
                                                                    (KeySelector<Long, Long>)
                                                                            t2 -> t2)
                                                            .window(EndOfStreamWindows.get())
                                                            .apply(
                                                                    new RichCoGroupFunction<
                                                                            Long, Long, Long>() {
                                                                        @Override
                                                                        public void coGroup(
                                                                                Iterable<Long>
                                                                                        iterable,
                                                                                Iterable<Long>
                                                                                        iterable1,
                                                                                Collector<Long>
                                                                                        collector) {
                                                                            System.out.println(
                                                                                    getRuntimeContext()
                                                                                            .getClass()
                                                                                            .getSimpleName());
                                                                            List<Long> b =
                                                                                    getRuntimeContext()
                                                                                            .getBroadcastVariable(
                                                                                                    "broadcast");
                                                                            System.out.println(
                                                                                    "broadcast variable: "
                                                                                            + b);
                                                                            for (Long iter :
                                                                                    iterable) {
                                                                                if (iter == null) {
                                                                                    continue;
                                                                                }
                                                                                collector.collect(
                                                                                        iter);
                                                                                System.out.println(
                                                                                        "task id : "
                                                                                                + getRuntimeContext()
                                                                                                        .getIndexOfThisSubtask()
                                                                                                + " value: "
                                                                                                + iter);
                                                                            }
                                                                            for (Long iter :
                                                                                    iterable1) {
                                                                                if (iter == null) {
                                                                                    continue;
                                                                                }
                                                                                System.out.println(
                                                                                        "task id : "
                                                                                                + getRuntimeContext()
                                                                                                        .getIndexOfThisSubtask()
                                                                                                + " value: "
                                                                                                + iter);
                                                                                collector.collect(
                                                                                        iter);
                                                                            }
                                                                        }
                                                                    });
                                                });

                                // output should be: 0 0 1 1 2 2 3 3 4 4 5 5
                                // and empty
                                return DataStreamList.of(
                                        coResult.filter(
                                                (FilterFunction<Long>)
                                                        longDenseVectorTuple2 ->
                                                                longDenseVectorTuple2 > 0L),
                                        coResult.filter(
                                                (FilterFunction<Long>)
                                                        longDenseVectorTuple2 ->
                                                                longDenseVectorTuple2 < 0L));
                            });

            DataStream<Integer> terminationCriteria =
                    feedbackVariableStream
                            .get(0)
                            .flatMap(new TerminateOnMaxIter2(2))
                            .returns(Types.INT);

            return new IterationBodyResult(
                    feedbackVariableStream, variableStreams, terminationCriteria);
        }
    }

    @Test
    public void testCoGroupWithIterationAndBroadcastLong() throws Exception {
        env.setParallelism(4);
        DataStream<Long> broadcast =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 5L), Types.LONG);
        DataStream<Long> dataStream1 =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 5L), Types.LONG);
        DataStream<Long> dataStream2 =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 5L), Types.LONG);
        DataStreamList coResult =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(dataStream1, dataStream2),
                        ReplayableDataStreamList.replay(broadcast),
                        IterationConfig.newBuilder()
                                .setOperatorLifeCycle(OperatorLifeCycle.PER_ROUND)
                                .build(),
                        new TrainIterationBodyWithBroadcastLong());

        List<Integer> counts = IteratorUtils.toList(coResult.get(0).executeAndCollect());
        System.out.println(counts.size());
    }

    @Test
    public void testIterationWithBroadcast() throws Exception {
        DataStream<Long> broadcast =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 5L), Types.LONG);
        DataStream<Long> input =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 5L), Types.LONG);
        DataStreamList coResult =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(input),
                        ReplayableDataStreamList.replay(broadcast),
                        IterationConfig.newBuilder()
                                .setOperatorLifeCycle(OperatorLifeCycle.PER_ROUND)
                                .build(),
                        new IterationBodyWithBroadcast());

        List<Integer> counts = IteratorUtils.toList(coResult.get(0).executeAndCollect());
        System.out.println(counts.size());
    }

    private static class IterationBodyWithBroadcast implements IterationBody {

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {

            // DataStreamList feedbackVariableStream =
            //        IterationBody.forEachRound(
            //                dataStreams,
            //                input -> {
            DataStream<Long> dataStream1 = variableStreams.get(0);

            DataStream<Long> coResult =
                    BroadcastUtils.withBroadcastStream(
                                    Arrays.asList(dataStream1),
                                    Collections.singletonMap("broadcast", dataStreams.get(0)),
                                    inputList -> {
                                        DataStream<Long> data1 =
                                                (DataStream<Long>) inputList.get(0);

                                        return data1.map(
                                                new MapFunction<Long, Long>() {
                                                    @Override
                                                    public Long map(Long longDenseVectorTuple2)
                                                            throws Exception {
                                                        return longDenseVectorTuple2;
                                                    }
                                                });
                                    })
                            .map(x -> x)
                            .returns(Types.LONG);

            //    return DataStreamList.of(
            //            coResult.filter(
            //                    (FilterFunction<Long>)
            //                            longDenseVectorTuple2 ->
            //                                    longDenseVectorTuple2 > 0L));
            // });

            DataStream<Integer> terminationCriteria =
                    coResult.<Long>flatMap(new TerminateOnMaxIter(2)).returns(Types.INT);

            return new IterationBodyResult(
                    DataStreamList.of(coResult), variableStreams, terminationCriteria);
        }
    }

    public static class TerminateOnMaxIter2
            implements IterationListener<Integer>, FlatMapFunction<Object, Integer> {

        private final int maxIter;

        public TerminateOnMaxIter2(Integer maxIter) {
            this.maxIter = maxIter;
        }

        @Override
        public void flatMap(Object value, Collector<Integer> out) {}

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<Integer> collector) {
            System.out.println("epoch watermark is: " + epochWatermark);
            if ((epochWatermark + 1) < maxIter) {
                collector.collect(0);
            }
        }

        @Override
        public void onIterationTerminated(Context context, Collector<Integer> collector) {}
    }

    @Test
    public void testKeyedWithBroadcast() throws Exception {
        env.setParallelism(2);
        DataStream<Long> broadcast =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 2L), Types.LONG);
        DataStream<Long> dataStream1 =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 5L), Types.LONG);
        DataStream<Long> result =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(dataStream1),
                        Collections.singletonMap("bc", broadcast),
                        inputList -> {
                            DataStream<Long> input = (DataStream<Long>) inputList.get(0);
                            DataStream<Long> output =
                                    DataStreamUtils.reduce(
                                            input.keyBy((KeySelector<Long, Long>) x -> x % 2),
                                            new MyReduceFunc());
                            return output;
                        });
        result.addSink(
                new SinkFunction<Long>() {
                    @Override
                    public void invoke(Long value) throws Exception {
                        SinkFunction.super.invoke(value);
                        System.out.println(value);
                    }
                });
        env.execute();
    }

    private static class MyReduceFunc extends RichReduceFunction<Long> {

        @Override
        public Long reduce(Long aLong, Long t1) throws Exception {
            Long x = (Long) getRuntimeContext().getBroadcastVariable("bc").get(0);
            System.out.println("bs" + x);
            return aLong + t1;
        }
    }

    @Test
    public void testPhysicalTransformation() {
        env.setParallelism(2);
        DataStream<Long> broadcast =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 2L), Types.LONG);
        KeyedStream<Long, Long> dataStream1 =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 5L), Types.LONG)
                        .keyBy(x -> x);
        System.out.println(broadcast.getTransformation() instanceof PhysicalTransformation);
        System.out.println(dataStream1.getTransformation() instanceof PhysicalTransformation);
    }
}
