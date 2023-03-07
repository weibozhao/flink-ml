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

package org.apache.flink.ml.recommendation.als;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichCoGroupFunction;
import org.apache.flink.api.common.functions.RichFilterFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.typeutils.GenericTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationConfig.OperatorLifeCycle;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.EndOfStreamWindows;
import org.apache.flink.ml.common.iteration.TerminateOnMaxIter;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.NormalEquationSolver;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;

import com.ibm.icu.impl.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * An Estimator which implements the linear regression algorithm.
 *
 * <p>See https://en.wikipedia.org/wiki/Linear_regression.
 */
public class Als implements Estimator<Als, AlsModel>, AlsParams<Als> {
    private static final Logger LOG = LoggerFactory.getLogger(Als.class);
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public Als() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public AlsModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> trainData = tEnv.toDataStream(inputs[0]);

        final String userCol = getUserCol();
        final String itemCol = getItemCol();
        final String ratingCol = getRatingCol();
        DataStream<Tuple3<Long, Long, Float>> alsInput =
                trainData.map(
                        (MapFunction<Row, Tuple3<Long, Long, Float>>)
                                value -> {
                                    Number user = value.getFieldAs(userCol);
                                    Number item = value.getFieldAs(itemCol);
                                    Number rating =
                                            ratingCol == null ? 0.0f : value.getFieldAs(ratingCol);

                                    return new Tuple3<>(
                                            user.longValue(),
                                            item.longValue(),
                                            rating.floatValue());
                                }).returns(Types.TUPLE(Types.LONG, Types.LONG, Types.FLOAT));

        DataStream<Ratings> graphData = initGraph(alsInput);
        DataStream<Factors> userItemFactors = initFactors(graphData, getRank(), getSeed());

        SingleOutputStreamOperator dataProfile = generateDataProfile(graphData,getRank(), getNumItemBlocks());

        DataStreamList result =
            Iterations.iterateBoundedStreamsUntilTermination(
                DataStreamList.of(userItemFactors),
                ReplayableDataStreamList.replay(graphData, dataProfile),
                IterationConfig.newBuilder()
                    .setOperatorLifeCycle(OperatorLifeCycle.PER_ROUND)
                    .build(),
                new TrainIterationBody(
                    getNumUserBlocks(), getRank(), getNonnegative(), getMaxIter()));

        DataStream<AlsModelData> modelData =
                DataStreamUtils.mapPartition(
                    result.get(0),
                    new MapPartitionFunction <List<Factors>, AlsModelData>() {
                        @Override
                        public void mapPartition(Iterable <List<Factors>> iterable, Collector <AlsModelData> collector) {
                            List<Tuple2<Long, float[]>> userFactors = new ArrayList<>();
                            List<Tuple2<Long, float[]>> itemFactors = new ArrayList<>();
                            for (List<Factors> factorsArray : iterable) {
                                for (Factors factors : factorsArray) {
                                    if (factors.identity == 0) {
                                        userFactors.add(
                                            Tuple2.of(factors.nodeId, factors.factors));
                                    } else {
                                        itemFactors.add(
                                            Tuple2.of(factors.nodeId, factors.factors));
                                    }
                                }
                            }
                            collector.collect(new AlsModelData(userFactors, itemFactors));
                            System.out.println("compute out OK.");
                        }
                    });
        modelData.getTransformation().setParallelism(1);

        AlsModel model = new AlsModel().setModelData(tEnv.fromDataStream(modelData));
        ReadWriteUtils.updateExistingParams(model, paramMap);
        return model;
    }

    private static class TrainIterationBody implements IterationBody {
        private final int numMiniBatches;
        private final int numFactors;
        private final boolean nonnegative;
        private final int numIters;

        public TrainIterationBody(
                int numMiniBatches, int numFactors, boolean nonnegative, int numIters) {
            this.numMiniBatches = numMiniBatches;
            this.numFactors = numFactors;
            this.nonnegative = nonnegative;
            this.numIters = numIters;
        }

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            final OutputTag <List<Factors>> modelDataOutputTag =
                new OutputTag<List<Factors>>("MODEL_OUTPUT") {};

            DataStream<Factors> userAndItemFactors = variableStreams.get(0);

            SingleOutputStreamOperator<Integer> iterationController =
                userAndItemFactors
                    .transform(
                        "iterationController",
                        Types.INT,
                        new IterationControllerFunc(modelDataOutputTag));

            DataStreamList feedbackVariableStream =
                    IterationBody.forEachRound(
                            dataStreams,
                            input -> {
                                DataStream<Ratings> graphData = dataStreams.get(0);
                                DataStream<DataProfile> dataProfile = dataStreams.get(1);
                                DataStream<Factors>
                                        factors =
                                                updateFactors(
                                                        userAndItemFactors,
                                                        graphData,
                                                        dataProfile,
                                                        numMiniBatches,
                                                        numFactors,
                                                        nonnegative,
                                                        iterationController);
                                return DataStreamList.of(factors);
                            });

            SingleOutputStreamOperator terminationCriteria =
                    iterationController.flatMap(new TerminateOnMaxIter(numIters));

            return new IterationBodyResult(
                    feedbackVariableStream,
                DataStreamList.of(
                    iterationController.getSideOutput(modelDataOutputTag)),
                terminationCriteria);
        }
    }

    private DataStream<Factors> initFactors(
            DataStream<Ratings> graphData, int rank, final long seed) {
        return graphData
                .map(
                        new RichMapFunction<Ratings, Factors>() {
                            transient Random random;
                            transient Factors reusedFactors;

                            @Override
                            public void open(Configuration parameters) {
                                random =
                                        new Random(
                                                getRuntimeContext().getIndexOfThisSubtask() + seed);
                                reusedFactors = new Factors();
                                reusedFactors.factors = new float[rank];
                            }

                            @Override
                            public Factors map(Ratings value) {
                                reusedFactors.identity = value.identity;
                                reusedFactors.nodeId = value.nodeId;
                                for (int i = 0; i < rank; i++) {
                                    reusedFactors.factors[i] = random.nextFloat();
                                }
                                return reusedFactors;
                            }
                        })
                .name("InitFactors");
    }

    private DataStream<Ratings> initGraph(DataStream<Tuple3<Long, Long, Float>> alsInput) {

        return alsInput.flatMap(
                        new RichFlatMapFunction<
                                Tuple3<Long, Long, Float>, Tuple4<Long, Long, Float, Byte>>() {

                            @Override
                            public void flatMap(
                                    Tuple3<Long, Long, Float> value,
                                    Collector<Tuple4<Long, Long, Float, Byte>> out) {
                                out.collect(Tuple4.of(value.f0, value.f1, value.f2, (byte) 0));
                                out.collect(Tuple4.of(value.f1, value.f0, value.f2, (byte) 1));
                            }
                        })
                .keyBy(new KeySelector <Tuple4<Long, Long, Float, Byte>, Pair<Byte, Long>>() {
                           @Override
                           public Pair <Byte, Long> getKey(Tuple4 <Long, Long, Float, Byte> value) {
                               return Pair.of(value.f3, value.f0);
                           }
                       })
                .window(EndOfStreamWindows.get())
                .process(
                        new ProcessWindowFunction<
                                Tuple4<Long, Long, Float, Byte>,
                                Ratings,
                                Pair<Byte, Long>,
                                TimeWindow>() {

                            @Override
                            public void process(
                                    Pair<Byte, Long> o,
                                    Context context,
                                    Iterable<Tuple4<Long, Long, Float, Byte>> iterable,
                                    Collector<Ratings> collector) throws InterruptedException {
                                byte identity = -1;
                                long srcNodeId = -1L;
                                List<Long> neighbors = new ArrayList<>();
                                List<Float> ratings = new ArrayList<>();

                                for (Tuple4<Long, Long, Float, Byte> v : iterable) {
                                    identity = v.f3;
                                    srcNodeId = v.f0;
                                    neighbors.add(v.f1);
                                    ratings.add(v.f2);
                                }

                                Ratings r = new Ratings();
                                r.nodeId = srcNodeId;
                                r.identity = identity;
                                r.neighbors = new long[neighbors.size()];
                                r.ratings = new float[neighbors.size()];

                                for (int i = 0; i < r.neighbors.length; i++) {
                                    r.neighbors[i] = neighbors.get(i);
                                    r.ratings[i] = ratings.get(i);
                                }
                                collector.collect(r);
                            }
                        }).returns(GenericTypeInfo.of(Ratings.class))
                .name("init_graph");
    }

    private static class IterationControllerFunc extends AbstractStreamOperator <Integer>
        implements OneInputStreamOperator <Factors, Integer>,
        IterationListener <Integer> {
        private final OutputTag <List<Factors>> modelDataOutputTag;
        private final List<Factors> factors = new ArrayList <>();
        public IterationControllerFunc(OutputTag <List<Factors>> modelDataOutputTag) {
            this.modelDataOutputTag = modelDataOutputTag;
        }

        @Override
        public void onEpochWatermarkIncremented(int epochWatermark, Context context, Collector <Integer> collector) {
            collector.collect(epochWatermark);
            System.out.println("epochWatermark : " + epochWatermark);
        }

        @Override
        public void onIterationTerminated(Context context, Collector <Integer> collector) {
            context.output(modelDataOutputTag, factors);
        }

        @Override
        public void processElement(StreamRecord <Factors> streamRecord) throws Exception {
            factors.add(streamRecord.getValue());
        }
    }

    /**
     * Update user factors or item factors in an iteration step. Only a mini-batch of users' or
     * items' factors are updated at one step.
     *
     * @param userAndItemFactors Users' and items' factors at the beginning of the step.
     * @param graphData Users' and items' ratings.
     * @param minBlocks Minimum number of mini-batches.
     * @param numFactors Number of factors.
     * @param nonnegative Whether to enforce non-negativity constraint.
     * @return Tuple2 of all factors and stop criterion.
     */
    private static DataStream<Factors> updateFactors(
            DataStream<Factors> userAndItemFactors,
            DataStream<Ratings> graphData,
            DataStream<DataProfile> profile,
            final int minBlocks,
            final int numFactors,
            final boolean nonnegative,
            SingleOutputStreamOperator<Integer> stepController) {

        Map<String, DataStream<?>> broadcastMap = new HashMap<>();
        broadcastMap.put("stepController", stepController);
        broadcastMap.put("profile", profile);

        // Get the mini-batch
        DataStream<Tuple2<Integer, Ratings>> miniBatch =
                BroadcastUtils.withBroadcastStream(
                                Collections.singletonList(graphData),
                                broadcastMap,
                                inputList -> {
                                    DataStream<Ratings> allData =
                                            (DataStream<Ratings>) inputList.get(0);

                                    return allData.filter(
                                            new RichFilterFunction<Ratings>() {
                                                private transient DataProfile profile;
                                                private transient int alsStepNo;
                                                private transient int userOrItem;
                                                private transient int subStepNo;
                                                private transient int numSubsteps;

                                                private int cnt = 0;
                                                @Override
                                                public void open(Configuration parameters) {
                                                    System.out.println(getRuntimeContext().getIndexOfThisSubtask() +  " generate mini batches.");
                                                    if (profile != null) {
                                                        subStepNo++;
                                                        if (userOrItem == 0) { // user step
                                                            if (subStepNo >= numSubsteps) {
                                                                userOrItem = 1;
                                                                numSubsteps = profile.numItemBatches;
                                                                subStepNo = 0;
                                                            }
                                                        } else if (userOrItem == 1) { // item step
                                                            if (subStepNo >= numSubsteps) {
                                                                userOrItem = 0;
                                                                numSubsteps = profile.numUserBatches;
                                                                subStepNo = 0;
                                                                alsStepNo++;
                                                            }
                                                        }
                                                        System.out.println( "ALS step no {}, user or item {}, sbu step no {}");
                                                        LOG.info(
                                                            "ALS step no {}, user or item {}, sbu step no {}",
                                                            alsStepNo,
                                                            userOrItem,
                                                            subStepNo);
                                                    }
                                                }

                                                @Override
                                                public void close() throws Exception {
                                                    super.close();
                                                    System.out.println(getRuntimeContext().getIndexOfThisSubtask() +  " generate mini batches end. "  + cnt);
                                                }

                                                @Override
                                                public boolean filter(Ratings value) {
                                                    cnt ++;
                                                    if (profile == null) {
                                                        profile =
                                                            (DataProfile)
                                                                getRuntimeContext()
                                                                    .getBroadcastVariable(
                                                                        "profile")
                                                                    .get(0);
                                                        LOG.info(
                                                            "Data profile : numItemBatches = {}, numUserBatches = {},"
                                                                + " numSamples = {}, numUsers = {}, numItems = {}, parallelism = {}",
                                                            profile.numItemBatches,
                                                            profile.numUserBatches,
                                                            profile.numSamples,
                                                            profile.numUsers,
                                                            profile.numItems,
                                                            profile.parallelism);
                                                        subStepNo = -1;
                                                        if (userOrItem == 1) {
                                                            userOrItem = 0;
                                                        } else
                                                        alsStepNo = 0;
                                                        numSubsteps = profile.numUserBatches;
                                                    }
                                                    List<Object> objs = getRuntimeContext().getBroadcastVariable("stepController");
                                                    //return true; // TODO : generate miniBatch for iteration
                                                    if (objs.size() == 0) { // todo : check stop condition of iteration.
                                                        return false;
                                                    } else {
                                                        return true;
                                                    }
                                                    //return alsStepNo < numIters
                                                    //        && value.nodeId == userOrItem
                                                    //        && Math.abs(value.identity) % numSubsteps
                                                    //                == subStepNo;
                                                }
                                            });
                                })
                        .map(
                                new RichMapFunction<Ratings, Tuple2<Integer, Ratings>>() {
                                    transient int partitionId;

                                    @Override
                                    public void open(Configuration parameters) {
                                        System.out.println(getRuntimeContext().getIndexOfThisSubtask() +  " Ratings map function...");
                                        this.partitionId =
                                                getRuntimeContext().getIndexOfThisSubtask();
                                    }

                                    @Override
                                    public void close() throws Exception {
                                        super.close();
                                        System.out.println(getRuntimeContext().getIndexOfThisSubtask() +  " Ratings map function end.");
                                    }

                                    @Override
                                    public Tuple2<Integer, Ratings> map(Ratings value) {
                                        return Tuple2.of(partitionId, value);
                                    }
                                });

        // Generate the request.
        // Tuple: srcPartitionId, targetIdentity, targetNodeId
        DataStream<Tuple3<Integer, Byte, Long>> request =
                miniBatch // Tuple: partitionId, ratings
                        .flatMap(
                                new RichFlatMapFunction<
                                        Tuple2<Integer, Ratings>, Tuple3<Integer, Byte, Long>>() {
                                    @Override
                                    public void open(Configuration parameters) throws Exception {
                                        System.out.println(getRuntimeContext().getIndexOfThisSubtask() +  " generate request...");
                                        super.open(parameters);
                                    }

                                    @Override
                                    public void close() throws Exception {
                                        super.close();
                                        System.out.println(getRuntimeContext().getIndexOfThisSubtask() +  " generate request end.");
                                    }

                                    @Override
                                    public void flatMap(
                                            Tuple2<Integer, Ratings> value,
                                            Collector<Tuple3<Integer, Byte, Long>> out) {
                                        int targetIdentity = 1 - value.f1.identity;
                                        int srcPartitionId = value.f0;
                                        long[] neighbors = value.f1.neighbors;
                                        for (long neighbor : neighbors) {
                                            out.collect(
                                                    Tuple3.of(
                                                            srcPartitionId,
                                                            (byte) targetIdentity,
                                                            neighbor));
                                        }
                                    }
                                })
                        .name("GenerateRequest");

        // Generate the response
        // Tuple: srcPartitionId, targetFactors
        DataStream<Tuple2<Integer, Factors>> response =
                request // Tuple: srcPartitionId, targetIdentity, targetNodeId
                        .coGroup(userAndItemFactors) // Factors
                        .where(
                                new KeySelector<Tuple3<Integer, Byte, Long>, Tuple2<Byte, Long>>() {

                                    @Override
                                    public Tuple2<Byte, Long> getKey(
                                            Tuple3<Integer, Byte, Long> value) {
                                        return Tuple2.of(value.f1, value.f2);
                                    }
                                })
                        .equalTo(
                                new KeySelector<Factors, Tuple2<Byte, Long>>() {

                                    @Override
                                    public Tuple2<Byte, Long> getKey(Factors value) {
                                        return Tuple2.of(value.identity, value.nodeId);
                                    }
                                })
                        .window(EndOfStreamWindows.get())
                        .apply(
                                new RichCoGroupFunction<
                                        Tuple3<Integer, Byte, Long>,
                                        Factors,
                                        Tuple2<Integer, Factors>>() {

                                    private transient int[] flag = null;
                                    private transient int[] partitionsIds = null;

                                    @Override
                                    public void open(Configuration parameters) {
                                        System.out.println(getRuntimeContext().getIndexOfThisSubtask() +  " generate response...");
                                        int numTasks =
                                                getRuntimeContext().getNumberOfParallelSubtasks();
                                        flag = new int[numTasks];
                                        partitionsIds = new int[numTasks];
                                    }

                                    @Override
                                    public void close() {
                                        System.out.println(getRuntimeContext().getIndexOfThisSubtask() +  " generate response end.");
                                        flag = null;
                                        partitionsIds = null;
                                    }

                                    @Override
                                    public void coGroup(
                                            Iterable<Tuple3<Integer, Byte, Long>> request,
                                            Iterable<Factors> factorsStore,
                                            Collector<Tuple2<Integer, Factors>> out) {
                                        System.out.println("response cogroup");

                                        if (request == null) {
                                            return;
                                        }

                                        int numRequests = 0;
                                        byte targetIdentity = -1;
                                        long targetNodeId = Long.MIN_VALUE;
                                        int numPartitionsIds = 0;
                                        Arrays.fill(flag, 0);

                                        // loop over request: srcBlockId, targetIdentity,
                                        // targetNodeId
                                        for (Tuple3<Integer, Byte, Long> v : request) {
                                            numRequests++;
                                            targetIdentity = v.f1;
                                            targetNodeId = v.f2;
                                            int partId = v.f0;
                                            if (flag[partId] == 0) {
                                                partitionsIds[numPartitionsIds++] = partId;
                                                flag[partId] = 1;
                                            }
                                        }

                                        if (numRequests == 0) {
                                            return;
                                        }

                                        for (Factors factors : factorsStore) {
                                            assert (factors.identity == targetIdentity
                                                    && factors.nodeId == targetNodeId);
                                            for (int i = 0; i < numPartitionsIds; i++) {
                                                int b = partitionsIds[i];
                                                out.collect(Tuple2.of(b, factors));
                                            }
                                        }
                                    }
                                }); // .name("GenerateResponse");

        DataStream<Factors> updatedBatchFactors;

        // Calculate factors
        // if (getImplicitprefs()) { todo:
        boolean implicitprefs = true;
        double regParam = 0.1;
        double alpha = 0.1;
        if (implicitprefs) {
            DataStream<double[]> yty = computeYtY(userAndItemFactors, numFactors, minBlocks);

            Map<String, DataStream<?>> ytyBroad = new HashMap<>();
            ytyBroad.put("stepController", stepController);
            ytyBroad.put("YtY", yty);
            // Tuple: Identity, nodeId, factors
            updatedBatchFactors =
                    BroadcastUtils.withBroadcastStream(
                            Arrays.asList(miniBatch, response),
                            ytyBroad,
                            inputList -> {
                                DataStream<Tuple2<Integer, Ratings>> data1 =
                                        (DataStream<Tuple2<Integer, Ratings>>) inputList.get(0);
                                DataStream<Tuple2<Integer, Factors>> data2 =
                                        (DataStream<Tuple2<Integer, Factors>>) inputList.get(1);
                                // Tuple: partitioId, Ratings
                                return data1.coGroup(data2) // Tuple: partitionId, Factors
                                        .where(value -> value.f0)
                                        .equalTo(value -> value.f0)
                                        .window(EndOfStreamWindows.get())
                                        .apply(
                                                new UpdateFactorsFunc(
                                                        false,
                                                        numFactors,
                                                        regParam,
                                                        alpha,
                                                        nonnegative));
                            });
        } else {
            // Tuple: Identity, nodeId, factors
            updatedBatchFactors =
                    miniBatch // Tuple: partitioId, Ratings
                            .coGroup(response) // Tuple: partitionId, Factors
                            .where(value -> value.f0)
                            .equalTo(value -> value.f0)
                            .window(EndOfStreamWindows.get())
                            .apply(
                                    new UpdateFactorsFunc(
                                            true,
                                            numFactors,
                                            regParam,
                                            nonnegative)); // .name("CalculateNewFactorsExplicit");
        }

        return userAndItemFactors
                        .coGroup(updatedBatchFactors)
                        .where(new KeySelector <Factors, Tuple2<Byte, Long>>() {
                                    @Override
                                    public Tuple2 <Byte, Long> getKey(Factors factors) throws Exception {
                                        return Tuple2.of(factors.identity, factors.nodeId);
                                    }
                                })
                        .equalTo(new KeySelector <Factors, Tuple2<Byte, Long>>() {
                                @Override
                                public Tuple2 <Byte, Long> getKey(Factors factors) throws Exception {
                                    return Tuple2.of(factors.identity, factors.nodeId);
                                }
                            })
                        .window(EndOfStreamWindows.get())
                        .apply(
                                new RichCoGroupFunction<Factors, Factors, Factors>() {

                                    @Override
                                    public void coGroup(
                                            Iterable<Factors> old,
                                            Iterable<Factors> updated,
                                            Collector<Factors> out) {

                                        assert (old != null);
                                        Iterator<Factors> iterator;

                                        if (updated == null
                                                || !(iterator = updated.iterator()).hasNext()) {
                                            for (Factors oldFactors : old) {
                                                out.collect(oldFactors);
                                            }
                                        } else {
                                            Factors newFactors = iterator.next();
                                            for (Factors oldFactors : old) {
                                                assert (oldFactors.identity == newFactors.identity
                                                        && oldFactors.nodeId == newFactors.nodeId);
                                                out.collect(newFactors);
                                            }
                                        }
                                    }
                                }); // .name("UpdateFactors");

        //DataStream<Integer> stopCriterion =
        //        BroadcastUtils.withBroadcastStream(
        //                Collections.singletonList(profile),
        //                Collections.singletonMap("stepController", stepController),
        //                inputList -> {
        //                    DataStream<DataProfile> initPro =
        //                            (DataStream<DataProfile>) inputList.get(0);
        //                    return initPro.flatMap(
        //                            new RichFlatMapFunction<DataProfile, Integer>() {
        //
        //                                @Override
        //                                public void flatMap(
        //                                        DataProfile pf, Collector<Integer> out) {
        //                                    int stepController =
        //                                        (int)getRuntimeContext().getBroadcastVariable("stepController").get(0);
        //                                    if (stepController < 2) {
        //                                            //< (pf.numUserBatches + pf.numItemBatches)
        //                                            //        * numIters) {
        //                                        out.collect(0);
        //                                    }
        //                                }
        //                            });
        //                });
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static Als load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /** Data profile. */
    public static class DataProfile implements Serializable {
        public long parallelism;
        public long numSamples;
        public long numUsers;
        public long numItems;

        public int numUserBatches;
        public int numItemBatches;

        // to make it POJO
        public DataProfile() {}

        void decideNumMiniBatches(int numFactors, int parallelism, int minBlocks) {
            this.numUserBatches =
                    decideUserMiniBatches(numSamples, numItems, numFactors, parallelism, minBlocks);
            this.numItemBatches =
                    decideUserMiniBatches(numSamples, numUsers, numFactors, parallelism, minBlocks);
        }

        static int decideUserMiniBatches(
                long numSamples, long numItems, int numFactors, int parallelism, int minBlocks) {
            final long taskCapacity = 2L /* nodes in million */ * 1024 * 1024 * 100 /* rank */;
            long numBatches = 1L;
            if (numItems * numFactors > taskCapacity) {
                numBatches = numSamples * numFactors / (parallelism * taskCapacity) + 1;
            }
            numBatches = Math.max(numBatches, minBlocks);
            return (int) numBatches;
        }
    }

    private static SingleOutputStreamOperator generateDataProfile(
            DataStream<Ratings> graphData, final int numFactors, final int minBlocks) {

        DataStream<Tuple3<Long, Long, Long>> middleData =
                DataStreamUtils.mapPartition(
                        graphData,
                        new RichMapPartitionFunction<Ratings, Tuple3<Long, Long, Long>>() {
                            @Override
                            public void open(Configuration parameters) throws Exception {
                                super.open(parameters);
                                System.out.println("generate Data profile.");
                            }

                            @Override
                            public void mapPartition(
                                    Iterable<Ratings> values,
                                    Collector<Tuple3<Long, Long, Long>> out) {
                                long numUsers = 0L;
                                long numItems = 0L;
                                long numRatings = 0L;
                                for (Ratings ratings : values) {
                                    if (ratings.identity == 0) {
                                        numUsers++;
                                        numRatings += ratings.neighbors.length;
                                    } else {
                                        numItems++;
                                    }
                                   // System.out.println("graph data to middle data.");
                                }
                                out.collect(Tuple3.of(numUsers, numItems, numRatings));
                            }
                        });

        return DataStreamUtils.reduce(
                        middleData,
                        new ReduceFunction<Tuple3<Long, Long, Long>>() {

                            @Override
                            public Tuple3<Long, Long, Long> reduce(
                                    Tuple3<Long, Long, Long> value1,
                                    Tuple3<Long, Long, Long> value2) {
                                value1.f0 += value2.f0;
                                value1.f1 += value2.f1;
                                value1.f2 += value2.f2;
                                return value1;
                            }
                        })
                .map(
                        new RichMapFunction<Tuple3<Long, Long, Long>, DataProfile>() {

                            @Override
                            public DataProfile map(Tuple3<Long, Long, Long> value) {
                                int parallelism =
                                        getRuntimeContext().getNumberOfParallelSubtasks();
                                DataProfile profile = new DataProfile();
                                profile.parallelism = parallelism;
                                profile.numUsers = value.f0;
                                profile.numItems = value.f1;
                                profile.numSamples = value.f2;
                                System.out.println("decide mini batches.");
                                profile.decideNumMiniBatches(numFactors, parallelism, minBlocks);
                                return profile;
                            }
                        }).returns(TypeInformation.of(DataProfile.class))
                .name("data_profiling");
    }

    /**
     * Update users' or items' factors in the local partition, after all depending remote factors
     * have been collected to the local partition.
     */
    private static class UpdateFactorsFunc
            extends RichCoGroupFunction<
                    Tuple2<Integer, Ratings>, Tuple2<Integer, Factors>, Factors> {
        final int numFactors;
        final double lambda;
        final double alpha;
        final boolean explicit;
        final boolean nonnegative;

        private int numNodes = 0;
        private long numEdges = 0L;
        private long numNeighbors = 0L;
        private boolean firstStep = true;
        private transient double[] yty = null;

        UpdateFactorsFunc(boolean explicit, int numFactors, double lambda, boolean nonnegative) {
            this.explicit = explicit;
            this.numFactors = numFactors;
            this.lambda = lambda;
            this.alpha = 0.;
            this.nonnegative = nonnegative;
        }

        UpdateFactorsFunc(
                boolean explicit,
                int numFactors,
                double lambda,
                double alpha,
                boolean nonnegative) {
            this.explicit = explicit;
            this.numFactors = numFactors;
            this.lambda = lambda;
            this.alpha = alpha;
            this.nonnegative = nonnegative;
        }

        @Override
        public void open(Configuration parameters) {
            numNodes = 0;
            numEdges = 0;
            numNeighbors = 0L;
            System.out.println("update factors : " + numNeighbors);
        }

        @Override
        public void close() {
            System.out.println("neighbors end.");
            LOG.info(
                    "Updated factors, num nodes {}, num edges {}, recv neighbors {}",
                    numNodes,
                    numEdges,
                    numNeighbors);
        }

        @Override
        public void coGroup(
                Iterable<Tuple2<Integer, Ratings>> rows,
                Iterable<Tuple2<Integer, Factors>> factors,
                Collector<Factors> out) {
            if (firstStep) {
                if (!explicit) {
                    this.yty = (double[]) (getRuntimeContext().getBroadcastVariable("YtY").get(0));
                }
                firstStep = false;
            }
            assert (rows != null && factors != null);
            List<Tuple2<Integer, Factors>> cachedFactors = new ArrayList<>();
            Map<Long, Integer> index2pos = new HashMap<>();
            numNeighbors = 0;
            // loop over received factors
            for (Tuple2<Integer, Factors> factor : factors) {
                cachedFactors.add(factor);
                index2pos.put(factor.f1.nodeId, (int) numNeighbors);
                numNeighbors++;
            }

            NormalEquationSolver ls = new NormalEquationSolver(numFactors);
            DenseVector x = new DenseVector(numFactors); // the solution buffer
            DenseVector buffer = new DenseVector(numFactors); // buffers for factors
            System.out.println("cogroup update factors");
            // loop over local nodes
            for (Tuple2<Integer, Ratings> row : rows) {
                numNodes++;
                numEdges += row.f1.neighbors.length;
                // solve an lease square problem
                ls.reset();

                if (explicit) {
                    long[] nb = row.f1.neighbors;
                    float[] rating = row.f1.ratings;
                    for (int i = 0; i < nb.length; i++) {
                        long index = nb[i];
                        Integer pos = index2pos.get(index);
                        cachedFactors.get(pos).f1.getFactorsAsDoubleArray(buffer.values);
                        ls.add(buffer, rating[i], 1.0);
                    }
                    ls.regularize(nb.length * lambda);
                    ls.solve(x, nonnegative);
                    System.out.println(x);
                } else {
                    ls.merge(new DenseMatrix(numFactors, numFactors, yty)); // put the YtY

                    int numExplicit = 0;
                    long[] nb = row.f1.neighbors;
                    float[] rating = row.f1.ratings;
                    for (int i = 0; i < nb.length; i++) {
                        long index = nb[i];
                        Integer pos = index2pos.get(index);
                        float r = rating[i];
                        double c1 = 0.;
                        if (r > 0) {
                            numExplicit++;
                            c1 = alpha * r;
                        }
                        if (pos == null) {
                            System.out.println(cachedFactors + " " + pos + " " + index);
                        }
                        cachedFactors.get(pos).f1.getFactorsAsDoubleArray(buffer.values);
                        //System.out.println("OK");
                        ls.add(buffer, ((r > 0.0) ? (1.0 + c1) : 0.0), c1);
                    }
                    numExplicit = Math.max(numExplicit, 1);
                    ls.regularize(numExplicit * lambda);
                    ls.solve(x, nonnegative);
                    System.out.println(x);
                }

                Factors updated = new Factors();
                updated.identity = row.f1.identity;
                updated.nodeId = row.f1.nodeId;
                updated.copyFactorsFromDoubleArray(x.values);
                out.collect(updated);
            }
        }
    }

    private static DataStream<double[]> computeYtY(
            DataStream<Factors> factors, final int numFactors, final int numMiniBatch) {
        DataStream<double[]> middleData =
                DataStreamUtils.mapPartition(
                        factors,
                        new RichMapPartitionFunction<Factors, double[]>() {

                            @Override
                            public void mapPartition(
                                    Iterable<Factors> values, Collector<double[]> out) {
                                //  int stepController = getIterationRuntimeContext().getSuperstepNumber() - 1;
                                int stepController = 1;
                                int identity = (stepController / numMiniBatch) % 2; // updating 'Identity'
                                int dst = 1 - identity;

                                double[] blockYtY = new double[numFactors * numFactors];
                                Arrays.fill(blockYtY, 0.);

                                for (Factors v : values) {
                                    if (v.identity != dst) {
                                        continue;
                                    }

                                    float[] factors1 = v.factors;
                                    for (int i = 0; i < numFactors; i++) {
                                        for (int j = 0; j < numFactors; j++) {
                                            blockYtY[i * numFactors + j] +=
                                                    factors1[i] * factors1[j];
                                        }
                                    }
                                }
                                out.collect(blockYtY);
                                System.out.println("compute YtY OK.");
                            }
                        });
        return DataStreamUtils.reduce(
                middleData,
                new ReduceFunction<double[]>() {

                    @Override
                    public double[] reduce(double[] value1, double[] value2) {
                        int n2 = numFactors * numFactors;
                        double[] sum = new double[n2];
                        for (int i = 0; i < n2; i++) {
                            sum[i] = value1[i] + value2[i];
                        }
                        return sum;
                    }
                });
    }
}
