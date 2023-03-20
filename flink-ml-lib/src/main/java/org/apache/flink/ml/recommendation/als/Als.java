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

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichCoGroupFunction;
import org.apache.flink.api.common.functions.RichFilterFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.typeutils.GenericTypeInfo;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
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
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.NormalEquationSolver;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * An Estimator which implements the Als algorithm.
 *
 * <p>ALS tries to decompose a matrix R as R = X * Yt. Here X and Y are called factor matrices.
 * Matrix R is usually a sparse matrix representing ratings given from users to items. ALS tries to
 * find X and Y that minimize || R - X * Yt ||^2. This is done by iterations. At each step, X is
 * fixed and Y is solved, then Y is fixed and X is solved.
 *
 * <p>The algorithm is described in "Large-scale Parallel Collaborative Filtering for the Netflix
 * Prize, 2007". This algorithm also supports implicit preference model described in "Collaborative
 * Filtering for Implicit Feedback Datasets, 2008".
 */
public class Als implements Estimator<Als, AlsModel>, AlsParams<Als> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private static final Logger LOG = LoggerFactory.getLogger(Als.class);

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
                trainData
                        .map(
                                (MapFunction<Row, Tuple3<Long, Long, Float>>)
                                        value -> {
                                            Long user = value.getFieldAs(userCol);
                                            Long item = value.getFieldAs(itemCol);

                                            Number rating =
                                                    ratingCol == null
                                                            ? 0.0F
                                                            : value.getFieldAs(ratingCol);

                                            return new Tuple3<>(user, item, rating.floatValue());
                                        })
                        .name("generateInputALsData")
                        .returns(Types.TUPLE(Types.LONG, Types.LONG, Types.FLOAT));

        /* Initializes variables before iteration. */
        DataStream<Ratings> ratingData = initRatings(alsInput);
        DataStream<Factors> userItemFactors = initFactors(ratingData, getRank(), getSeed());
        DataStream yty = initYty(userItemFactors);

        /* The iterations to solve the als problem. */
        DataStream<List<Factors>> result =
                Iterations.iterateBoundedStreamsUntilTermination(
                                DataStreamList.of(userItemFactors, yty),
                                ReplayableDataStreamList.replay(ratingData),
                                IterationConfig.newBuilder()
                                        .setOperatorLifeCycle(OperatorLifeCycle.ALL_ROUND)
                                        .build(),
                                new TrainIterationBody(
                                        getRank(),
                                        getNonNegative(),
                                        getMaxIter(),
                                        getImplicitPrefs(),
                                        getRegParam(),
                                        getAlpha(),
                                        getNumUserBlocks(),
                                        getNumItemBlocks()))
                        .get(0);

        /* Generates model data with iteration results. */
        DataStream<AlsModelData> modelData =
                result.transform(
                                "generateModelData",
                                TypeInformation.of(AlsModelData.class),
                                new GenerateModelData())
                        .name("generateModelData")
                        .setParallelism(1);

        AlsModel model = new AlsModel().setModelData(tEnv.fromDataStream(modelData));
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    /**
     * The train iteration body includes two main actions. The first action is updating the user
     * factors with the rating information and the item factors. The second one is updating the item
     * factors with the rating information and the user factors. These two actions occur alternately
     * and iteratively.
     */
    private static class TrainIterationBody implements IterationBody {
        private final int numFactors;
        private final boolean nonNegative;
        private final int maxIter;
        private final boolean implicitPrefs;
        private final double regParam;
        private final double alpha;
        private final int numUserBlocks;
        private final int numItemBlocks;

        public TrainIterationBody(
                int numFactors,
                boolean nonNegative,
                int maxIter,
                boolean implicitPrefs,
                double regParam,
                double alpha,
                int numUserBlocks,
                int numItemBlocks) {
            this.numFactors = numFactors;
            this.nonNegative = nonNegative;
            this.maxIter = maxIter;
            this.implicitPrefs = implicitPrefs;
            this.regParam = regParam;
            this.alpha = alpha;
            this.numUserBlocks = numUserBlocks;
            this.numItemBlocks = numItemBlocks;
        }

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<Factors> userAndItemFactors = variableStreams.get(0);
            DataStream<Tuple2<double[], Integer>> yty = variableStreams.get(1);

            DataStreamList feedbackVariableStream =
                    IterationBody.forEachRound(
                            dataStreams,
                            input -> {
                                DataStream<Ratings> ratingData = dataStreams.get(0);
                                Tuple2<DataStream<Factors>, DataStream<Tuple2<double[], Integer>>>
                                        factorsAndYty =
                                                updateFactors(
                                                        userAndItemFactors,
                                                        ratingData,
                                                        yty,
                                                        numFactors,
                                                        nonNegative,
                                                        implicitPrefs,
                                                        regParam,
                                                        alpha,
                                                        numUserBlocks,
                                                        numItemBlocks);
                                return DataStreamList.of(factorsAndYty.f0, factorsAndYty.f1);
                            });

            DataStream<Factors> feedbackFactors = variableStreams.get(0);

            final OutputTag<List<Factors>> modelDataOutputTag =
                    new OutputTag<List<Factors>>("MODEL_OUTPUT") {};

            SingleOutputStreamOperator<Integer> iterationController =
                    feedbackFactors
                            .flatMap(
                                    new IterationControllerFunc(
                                            modelDataOutputTag,
                                            maxIter,
                                            numUserBlocks,
                                            numItemBlocks))
                            .name("iterationController");

            return new IterationBodyResult(
                    feedbackVariableStream,
                    DataStreamList.of(iterationController.getSideOutput(modelDataOutputTag)),
                    iterationController);
        }
    }

    /** Generates the ModelData from the results of iteration. */
    private static class GenerateModelData extends AbstractStreamOperator<AlsModelData>
            implements OneInputStreamOperator<List<Factors>, AlsModelData>, BoundedOneInput {

        private final List<Tuple2<Long, float[]>> userFactors = new ArrayList<>();
        private final List<Tuple2<Long, float[]>> itemFactors = new ArrayList<>();

        @Override
        public void endInput() throws Exception {
            LOG.info("generate model   ... " + System.currentTimeMillis());

            output.collect(new StreamRecord<>(new AlsModelData(userFactors, itemFactors)));
        }

        @Override
        public void processElement(StreamRecord<List<Factors>> streamRecord) throws Exception {
            List<Factors> factorsArray = streamRecord.getValue();

            for (Factors factors : factorsArray) {
                if (factors.identity == 0) {
                    userFactors.add(Tuple2.of(factors.nodeId, factors.factors));
                } else {
                    itemFactors.add(Tuple2.of(factors.nodeId, factors.factors));
                }
            }
        }
    }

    /** Generates the ModelData from the results of iteration. */
    private static class GenerateRequest extends AbstractStreamOperator<Tuple3<Integer, Byte, Long>>
            implements OneInputStreamOperator<Ratings, Tuple3<Integer, Byte, Long>>,
                    BoundedOneInput {

        private int srcPartitionId;
        private int targetIdentity;
        private final Set<Long> neighbors = new HashSet<>(100000);

        @Override
        public void endInput() throws Exception {
            LOG.info("generate request   ... " + System.currentTimeMillis());
            for (long neighbor : neighbors) {
                output.collect(
                        new StreamRecord<>(
                                Tuple3.of(srcPartitionId, (byte) targetIdentity, neighbor)));
            }
            // output.collect(new StreamRecord<>(new AlsModelData(userFactors, itemFactors)));
        }

        @Override
        public void processElement(StreamRecord<Ratings> streamRecord) throws Exception {
            Ratings value = streamRecord.getValue();
            targetIdentity = 1 - value.identity;
            srcPartitionId = getRuntimeContext().getIndexOfThisSubtask();
            long[] ns = value.neighbors;

            for (long val : ns) {
                neighbors.add(val);
            }
        }
    }

    /**
     * Initializes the user factors and the item factors with a random function.
     *
     * @param ratingData Rating data generated by graph.
     * @param rank Rank of the factorization.
     * @param seed The random seed.
     * @return The factors of the user and the item.
     */
    private DataStream<Factors> initFactors(
            DataStream<Ratings> ratingData, int rank, final long seed) {
        return ratingData
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

    /**
     * Initializes the ratings data with the input graph.
     *
     * @param alsInput The input graph.
     * @return The ratings data.
     */
    private DataStream<Ratings> initRatings(DataStream<Tuple3<Long, Long, Float>> alsInput) {

        DataStream<Ratings> ratings =
                alsInput.flatMap(
                                new RichFlatMapFunction<
                                        Tuple3<Long, Long, Float>,
                                        Tuple4<Long, Long, Float, Byte>>() {

                                    @Override
                                    public void flatMap(
                                            Tuple3<Long, Long, Float> value,
                                            Collector<Tuple4<Long, Long, Float, Byte>> out) {
                                        out.collect(
                                                Tuple4.of(value.f0, value.f1, value.f2, (byte) 0));
                                        out.collect(
                                                Tuple4.of(value.f1, value.f0, value.f2, (byte) 1));
                                    }
                                })
                        .keyBy(
                                (KeySelector<Tuple4<Long, Long, Float, Byte>, String>)
                                        value -> value.f3.toString() + value.f0)
                        .window(EndOfStreamWindows.get())
                        .process(
                                new ProcessWindowFunction<
                                        Tuple4<Long, Long, Float, Byte>,
                                        Ratings,
                                        String,
                                        TimeWindow>() {

                                    @Override
                                    public void process(
                                            String o,
                                            Context context,
                                            Iterable<Tuple4<Long, Long, Float, Byte>> iterable,
                                            Collector<Ratings> collector) {
                                        byte identity = -1;
                                        long srcNodeId = -1L;
                                        List<Tuple2<Long, Float>> neighbors = new ArrayList<>();

                                        for (Tuple4<Long, Long, Float, Byte> v : iterable) {
                                            identity = v.f3;
                                            srcNodeId = v.f0;
                                            neighbors.add(Tuple2.of(v.f1, v.f2));
                                        }
                                        Ratings returnRatings = new Ratings();
                                        returnRatings.nodeId = srcNodeId;
                                        returnRatings.identity = identity;
                                        returnRatings.neighbors = new long[neighbors.size()];
                                        returnRatings.ratings = new float[neighbors.size()];

                                        for (int i = 0; i < returnRatings.neighbors.length; i++) {
                                            returnRatings.neighbors[i] = neighbors.get(i).f0;
                                            returnRatings.ratings[i] = neighbors.get(i).f1;
                                        }
                                        collector.collect(returnRatings);
                                    }
                                })
                        .returns(GenericTypeInfo.of(Ratings.class))
                        .name("initRatings");

        return DataStreamUtils.mapPartition(ratings, new SortFunction());
    }

    private static class SortFunction implements MapPartitionFunction<Ratings, Ratings> {

        @Override
        public void mapPartition(Iterable<Ratings> iterable, Collector<Ratings> collector) {
            LOG.info("sort function running ... " + System.currentTimeMillis());

            List<Ratings> listRatings = new ArrayList<>();
            for (Ratings ratings : iterable) {
                listRatings.add(ratings);
            }
            listRatings.sort(
                    (o1, o2) -> {
                        if (o1.nodeId != o2.nodeId) {
                            return Long.compare(o1.nodeId, o2.nodeId);
                        } else {
                            return Byte.compare(o1.identity, o2.identity);
                        }
                    });
            for (Ratings ratings : listRatings) {
                collector.collect(ratings);
            }
            LOG.info("sort function end   ... " + System.currentTimeMillis());
        }
    }

    /** Gets the current step of the iteration. */
    private static class StepFunction extends AbstractStreamOperator<Integer>
            implements OneInputStreamOperator<Factors, Integer>,
                    BoundedOneInput,
                    IterationListener<Integer> {

        @Override
        public void endInput() throws Exception {
            LOG.info("calculate step    ... " + System.currentTimeMillis());

            output.collect(new StreamRecord<>(-1));
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<Integer> collector) {
            collector.collect(epochWatermark);
        }

        @Override
        public void onIterationTerminated(Context context, Collector<Integer> collector) {}

        @Override
        public void processElement(StreamRecord<Factors> streamRecord) throws Exception {}
    }

    /**
     * Update user factors or item factors in an iteration step. Only a mini-batch of users' or
     * items' factors are updated at one step.
     *
     * @param userAndItemFactors Users' and items' factors at the beginning of this step.
     * @param ratingData Users' and items' ratings.
     * @param yty Matrix variables used in implicit mode.
     * @param numFactors Number of factors.
     * @param nonNegative Whether to enforce non-negativity constraint.
     * @param implicitPrefs Is implicit mode or not.
     * @param regParam Regular parameter.
     * @param alpha Alpha parameter.
     * @param numUserBlocks Number of user blocks.
     * @param numItemBlocks Number of item blocks.
     * @return Tuple2 of all factors and yty matrix with epoch.
     */
    @SuppressWarnings({"unchecked"})
    private static Tuple2<DataStream<Factors>, DataStream<Tuple2<double[], Integer>>> updateFactors(
            DataStream<Factors> userAndItemFactors,
            DataStream<Ratings> ratingData,
            DataStream<Tuple2<double[], Integer>> yty,
            final int numFactors,
            final boolean nonNegative,
            final boolean implicitPrefs,
            final double regParam,
            final double alpha,
            final int numUserBlocks,
            final int numItemBlocks) {

        SingleOutputStreamOperator<Integer> step =
                userAndItemFactors
                        .transform("step", Types.INT, new StepFunction())
                        .name("stepGenerator");

        Map<String, DataStream<?>> broadcastMap = new HashMap<>();
        broadcastMap.put("step", step);

        // Gets the miniBatch
        DataStream<Ratings> miniBatch =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(ratingData),
                        broadcastMap,
                        inputList -> {
                            DataStream<Ratings> allData = (DataStream<Ratings>) inputList.get(0);

                            return allData.filter(
                                    new RichFilterFunction<Ratings>() {
                                        private int userOrItem;
                                        private int subStepNo = -1;
                                        private int numSubSteps;
                                        private boolean isFirst = true;

                                        @Override
                                        public void open(Configuration parameters)
                                                throws Exception {
                                            super.open(parameters);
                                            LOG.info(
                                                    "generate mini Batch begin   ... "
                                                            + System.currentTimeMillis());
                                        }

                                        @Override
                                        public void close() throws Exception {
                                            super.close();
                                            LOG.info(
                                                    "generate mini Batch end     ... "
                                                            + System.currentTimeMillis());
                                        }

                                        @Override
                                        public boolean filter(Ratings value) {
                                            if (isFirst) {
                                                LOG.info("generate mini batch processing ...");
                                                isFirst = false;
                                            }
                                            if (subStepNo == -1) {
                                                List<Object> broadStep =
                                                        getRuntimeContext()
                                                                .getBroadcastVariable("step");

                                                int step =
                                                        broadStep.size() > 0
                                                                ? (int) broadStep.get(0)
                                                                : -1;

                                                if (step == -1) {
                                                    subStepNo = -2;
                                                    userOrItem = -1;
                                                } else {
                                                    int superStep =
                                                            step % (numUserBlocks + numItemBlocks);

                                                    if (superStep < numUserBlocks) {
                                                        subStepNo = superStep;
                                                        userOrItem = 0;
                                                    } else {
                                                        subStepNo = superStep - numUserBlocks;
                                                        userOrItem = 1;
                                                    }
                                                }

                                                numSubSteps =
                                                        (userOrItem == 0)
                                                                ? numUserBlocks
                                                                : numItemBlocks;
                                            }

                                            return value.identity == userOrItem
                                                    && Math.abs(value.identity) % numSubSteps
                                                            == subStepNo;
                                        }
                                    });
                        });

        // Generates the request.
        // Tuple: srcPartitionId, targetIdentity, targetNodeId
        DataStream<Tuple3<Integer, Byte, Long>> request =
                miniBatch.transform(
                        "generateRequest",
                        new TupleTypeInfo<>(
                                BasicTypeInfo.INT_TYPE_INFO,
                                BasicTypeInfo.BYTE_TYPE_INFO,
                                BasicTypeInfo.LONG_TYPE_INFO),
                        new GenerateRequest());

        // Tuple: partitionId, ratings
        //    .flatMap(
        //            new RichFlatMapFunction<
        //                    Tuple2<Integer, Ratings>, Tuple3<Integer, Byte, Long>>() {
        //                private boolean isFirst = true;
        //
        //                @Override
        //                public void open(Configuration parameters) throws Exception {
        //                    super.open(parameters);
        //                    LOG.info(
        //                            "generate request begin   ... "
        //                                    + System.currentTimeMillis());
        //                }
        //
        //                @Override
        //                public void close() throws Exception {
        //                    super.close();
        //                    LOG.info(
        //                            "generate request end     ... "
        //                                    + System.currentTimeMillis());
        //                }
        //
        //                @Override
        //                public void flatMap(
        //                        Tuple2<Integer, Ratings> value,
        //                        Collector<Tuple3<Integer, Byte, Long>> out) {
        //                    if (isFirst) {
        //                        LOG.info("generate request batch processing ...");
        //                        isFirst = false;
        //                    }
        //                    int targetIdentity = 1 - value.f1.identity;
        //                    int srcPartitionId = value.f0;
        //                    long[] neighbors = value.f1.neighbors;
        //
        //                    for (long neighbor : neighbors) {
        //                        out.collect(
        //                                Tuple3.of(
        //                                        srcPartitionId,
        //                                        (byte) targetIdentity,
        //                                        neighbor));
        //                    }
        //                }
        //            })
        //   .name("generateRequest");

        // request =
        //        request.map(
        //                new MapFunction<
        //                        Tuple3<Integer, Byte, Long>, Tuple3<Integer, Byte, Long>>() {
        //                    @Override
        //                    public Tuple3<Integer, Byte, Long> map(
        //                            Tuple3<Integer, Byte, Long> integerByteLongTuple3)
        //                            throws Exception {
        //                        System.out.println(integerByteLongTuple3);
        //                        return integerByteLongTuple3;
        //                    }
        //                });
        DataStream<Factors> userOrItemFactors =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(userAndItemFactors),
                        broadcastMap,
                        inputList -> {
                            DataStream<Factors> allData = (DataStream<Factors>) inputList.get(0);
                            return allData.filter(
                                            new RichFilterFunction<Factors>() {
                                                private Integer subStep;

                                                @Override
                                                public void open(Configuration parameters)
                                                        throws Exception {
                                                    super.open(parameters);
                                                    LOG.info(
                                                            "filter userOrItem begin   ... "
                                                                    + System.currentTimeMillis());
                                                }

                                                @Override
                                                public void close() throws Exception {
                                                    super.close();
                                                    LOG.info(
                                                            "filter userOrItem end     ... "
                                                                    + System.currentTimeMillis());
                                                }

                                                @Override
                                                public boolean filter(Factors factors) {
                                                    if (subStep == null) {
                                                        LOG.info(
                                                                "generate user or item batch processing ...");
                                                        List<Object> broadStep =
                                                                getRuntimeContext()
                                                                        .getBroadcastVariable(
                                                                                "step");

                                                        int step =
                                                                broadStep.size() > 0
                                                                        ? (int) broadStep.get(0)
                                                                        : -1;
                                                        subStep =
                                                                step
                                                                        % (numUserBlocks
                                                                                + numItemBlocks);
                                                    }
                                                    if (subStep >= numUserBlocks) {
                                                        return factors.identity == 0;
                                                    } else {
                                                        return factors.identity == 1;
                                                    }
                                                }
                                            })
                                    .name("filterUserOrItem");
                        });

        /* Generates the response information, which will be used to update the factors. */
        DataStream<Tuple2<Integer, Factors>> response =
                request.coGroup(userOrItemFactors)
                        .where(
                                (KeySelector<Tuple3<Integer, Byte, Long>, String>)
                                        value -> value.f1.toString() + value.f2)
                        .equalTo(
                                (KeySelector<Factors, String>)
                                        value -> String.valueOf(value.identity) + value.nodeId)
                        .window(EndOfStreamWindows.get())
                        .apply(
                                new RichCoGroupFunction<
                                        Tuple3<Integer, Byte, Long>,
                                        Factors,
                                        Tuple2<Integer, Factors>>() {
                                    private boolean isFirst = true;
                                    private transient int[] flag = null;
                                    private transient int[] partitionsIds = null;

                                    @Override
                                    public void open(Configuration parameters) {
                                        LOG.info(
                                                "calculate response Function running ... "
                                                        + System.currentTimeMillis());
                                        int numTasks =
                                                getRuntimeContext().getNumberOfParallelSubtasks();
                                        flag = new int[numTasks];
                                        partitionsIds = new int[numTasks];
                                    }

                                    @Override
                                    public void close() {
                                        flag = null;
                                        partitionsIds = null;
                                        LOG.info(
                                                "calculate response Function end    ... "
                                                        + System.currentTimeMillis());
                                    }

                                    @Override
                                    public void coGroup(
                                            Iterable<Tuple3<Integer, Byte, Long>> request,
                                            Iterable<Factors> factorsStore,
                                            Collector<Tuple2<Integer, Factors>> out) {
                                        if (isFirst) {
                                            LOG.info(
                                                    "calculate response Function processing    ... "
                                                            + System.currentTimeMillis());
                                            isFirst = false;
                                        }
                                        if (!request.iterator().hasNext()
                                                || !factorsStore.iterator().hasNext()) {
                                            return;
                                        }

                                        int numRequests = 0;
                                        byte targetIdentity = -1;
                                        long targetNodeId = Long.MIN_VALUE;
                                        int numPartitionsIds = 0;
                                        Arrays.fill(flag, 0);

                                        /* loop over request: srcBlockId, targetIdentity, targetNodeId*/
                                        for (Tuple3<Integer, Byte, Long> t3 : request) {
                                            numRequests++;
                                            targetIdentity = t3.f1;
                                            targetNodeId = t3.f2;
                                            int partId = t3.f0;

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
                                });

        /* Repartition of the data is to improve the performance of coGroup. */
        response = response.partitionCustom((chunkId, numPartitions) -> chunkId, x -> x.f0);
        DataStream<Factors> updatedFactors;

        DataStream<Tuple2<double[], Integer>> newYty;
        // Calculates factors
        if (implicitPrefs) {
            newYty = computeYty(userOrItemFactors, yty, numFactors, numUserBlocks, numItemBlocks);
            // Tuple: Identity, nodeId, factors
            updatedFactors =
                    BroadcastUtils.withBroadcastStream(
                            Arrays.asList(miniBatch, response),
                            Collections.singletonMap(
                                    "Yty",
                                    newYty.map(
                                            (MapFunction<Tuple2<double[], Integer>, double[]>)
                                                    ytyWithEpoch -> ytyWithEpoch.f0)),
                            inputList -> {
                                DataStream<Ratings> miniBatchRatings =
                                        (DataStream<Ratings>) inputList.get(0);
                                DataStream<Tuple2<Integer, Factors>> responseData =
                                        (DataStream<Tuple2<Integer, Factors>>) inputList.get(1);

                                // Tuple: partitionId, Ratings
                                return miniBatchRatings
                                        .connect(responseData) // Tuple: partitionId, Factors
                                        .transform(
                                                "mapPartition",
                                                TypeInformation.of(Factors.class),
                                                new UpdateFactorsFunc(
                                                        false,
                                                        numFactors,
                                                        regParam,
                                                        alpha,
                                                        nonNegative));
                            });
        } else {
            newYty = yty;
            updatedFactors =
                    miniBatch
                            .connect(response)
                            .transform(
                                    "mapPartition",
                                    TypeInformation.of(Factors.class),
                                    new UpdateFactorsFunc(
                                            true, numFactors, regParam, 0.0, nonNegative));
        }

        DataStream<Factors> factors =
                userAndItemFactors
                        .coGroup(updatedFactors)
                        .where(
                                (KeySelector<Factors, String>)
                                        value -> String.valueOf(value.identity) + value.nodeId)
                        .equalTo(
                                (KeySelector<Factors, String>)
                                        value -> String.valueOf(value.identity) + value.nodeId)
                        .window(EndOfStreamWindows.get())
                        .apply(
                                new RichCoGroupFunction<Factors, Factors, Factors>() {
                                    @Override
                                    public void open(Configuration parameters) throws Exception {
                                        super.open(parameters);
                                        LOG.info(
                                                "co group to merge factors running ... "
                                                        + System.currentTimeMillis());
                                    }

                                    @Override
                                    public void close() throws Exception {
                                        super.close();
                                        LOG.info(
                                                "co group to merge factors end    ... "
                                                        + System.currentTimeMillis());
                                    }

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
                                });
        return Tuple2.of(factors, newYty);
    }

    private static class IterationControllerFunc
            implements FlatMapFunction<Factors, Integer>, IterationListener<Integer> {
        private final OutputTag<List<Factors>> modelDataOutputTag;
        private final int maxIter;
        private final int numUserBlocks;
        private final int numItemBlocks;

        private final List<Factors> factorsList = new ArrayList<>();

        public IterationControllerFunc(
                OutputTag<List<Factors>> modelDataOutputTag,
                int maxIter,
                int numUserBlocks,
                int numItemBlocks) {
            this.modelDataOutputTag = modelDataOutputTag;
            this.maxIter = maxIter;
            this.numUserBlocks = numUserBlocks;
            this.numItemBlocks = numItemBlocks;
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<Integer> collector) {
            if ((epochWatermark + 1) < maxIter * (numUserBlocks + numItemBlocks)) {
                collector.collect(epochWatermark);
            }
        }

        @Override
        public void onIterationTerminated(Context context, Collector<Integer> collector) {
            context.output(modelDataOutputTag, factorsList);
        }

        @Override
        public void flatMap(Factors factors, Collector<Integer> collector) throws Exception {
            factorsList.add(factors);
        }
    }

    private static class UpdateFactorsFunc extends AbstractStreamOperator<Factors>
            implements TwoInputStreamOperator<Ratings, Tuple2<Integer, Factors>, Factors>,
                    IterationListener<Factors> {
        Map<Long, Factors> bufferedFactors = new HashMap<>();
        List<Ratings> bufferedRatings = new ArrayList<>();

        final int numFactors;
        final double lambda;
        final double alpha;
        final boolean explicit;
        final boolean nonNegative;

        private transient double[] yty = null;

        UpdateFactorsFunc(
                boolean explicit,
                int numFactors,
                double lambda,
                double alpha,
                boolean nonNegative) {
            this.explicit = explicit;
            this.numFactors = numFactors;
            this.lambda = lambda;
            this.alpha = alpha;
            this.nonNegative = nonNegative;
        }

        @Override
        public void processElement1(StreamRecord<Ratings> streamRecord) {
            bufferedRatings.add(streamRecord.getValue());
        }

        @Override
        public void processElement2(StreamRecord<Tuple2<Integer, Factors>> streamRecord) {
            Factors fac = streamRecord.getValue().f1;
            bufferedFactors.put(fac.nodeId, fac);
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<Factors> collector) {
            if (!explicit) {
                yty = (double[]) getRuntimeContext().getBroadcastVariable("Yty").get(0);
            }
            NormalEquationSolver ls = new NormalEquationSolver(numFactors);
            DenseVector x = new DenseVector(numFactors);
            DenseVector buffer = new DenseVector(numFactors);
            /* loops over local nodes. */
            for (Ratings t2 : bufferedRatings) {
                /* solves an lease square problem. */
                ls.reset();

                if (explicit) {
                    long[] nb = t2.neighbors;
                    float[] rating = t2.ratings;
                    for (int i = 0; i < nb.length; i++) {
                        bufferedFactors.get(nb[i]).getFactorsAsDoubleArray(buffer.values);
                        ls.add(buffer, rating[i], 1.0);
                    }
                    ls.regularize(nb.length * lambda);
                    ls.solve(x, nonNegative);
                } else {
                    ls.merge(new DenseMatrix(numFactors, numFactors, yty));

                    int numExplicit = 0;
                    long[] nb = t2.neighbors;
                    float[] rating = t2.ratings;

                    for (int i = 0; i < nb.length; i++) {
                        float r = rating[i];
                        double c1 = 0.;

                        if (r > 0) {
                            numExplicit++;
                            c1 = alpha * r;
                        }

                        bufferedFactors.get(nb[i]).getFactorsAsDoubleArray(buffer.values);
                        ls.add(buffer, ((r > 0.0) ? (1.0 + c1) : 0.0), c1);
                    }

                    numExplicit = Math.max(numExplicit, 1);
                    ls.regularize(numExplicit * lambda);
                    ls.solve(x, nonNegative);
                }

                Factors updated = new Factors();
                updated.identity = t2.identity;
                updated.nodeId = t2.nodeId;
                updated.copyFactorsFromDoubleArray(x.values);
                output.collect(new StreamRecord<>(updated));
            }
        }

        @Override
        public void onIterationTerminated(Context context, Collector<Factors> collector) {}
    }

    private static DataStream<Tuple2<double[], Integer>> initYty(DataStream<Factors> factors) {
        StreamExecutionEnvironment env = factors.getExecutionEnvironment();
        List<Tuple2<double[], Integer>> data = new ArrayList<>();
        data.add(Tuple2.of(new double[0], -1));
        return env.fromCollection(
                data, new TupleTypeInfo<>(TypeInformation.of(double[].class), Types.INT));
    }

    private static DataStream<Tuple2<double[], Integer>> computeYty(
            DataStream<Factors> factors,
            DataStream<Tuple2<double[], Integer>> yty,
            final int numFactors,
            final int numUserBlocks,
            final int numItemBlocks) {

        DataStream<Tuple2<double[], Integer>> localYty =
                factors.flatMap(new ComputeLocalYty(numFactors, numUserBlocks, numItemBlocks));

        DataStream<Tuple2<double[], Integer>> newYty =
                DataStreamUtils.reduce(
                        localYty,
                        (ReduceFunction<Tuple2<double[], Integer>>)
                                (value1, value2) -> {
                                    int n2 = numFactors * numFactors;

                                    for (int j = 0; j < n2; ++j) {
                                        value1.f0[j] += value2.f0[j];
                                    }
                                    return value1;
                                });

        return yty.union(newYty)
                .flatMap(
                        new FlatMapFunction<
                                Tuple2<double[], Integer>, Tuple2<double[], Integer>>() {
                            private boolean firstStep = true;
                            private Tuple2<double[], Integer> ytyLast;

                            @Override
                            public void flatMap(
                                    Tuple2<double[], Integer> ytyWithEpoch,
                                    Collector<Tuple2<double[], Integer>> collector) {
                                if (firstStep) {
                                    ytyLast = ytyWithEpoch;
                                    firstStep = false;
                                } else {
                                    if (ytyLast.f1 > ytyWithEpoch.f1) {
                                        if (BLAS.norm2(new DenseVector(ytyLast.f0)) == 0) {
                                            collector.collect(ytyWithEpoch);
                                        } else {
                                            collector.collect(ytyLast);
                                        }
                                    } else {
                                        if (BLAS.norm2(new DenseVector(ytyWithEpoch.f0)) == 0) {
                                            collector.collect(ytyLast);
                                        } else {
                                            collector.collect(ytyWithEpoch);
                                        }
                                    }
                                }
                            }
                        })
                .setParallelism(1);
    }

    private static class ComputeLocalYty
            extends RichFlatMapFunction<Factors, Tuple2<double[], Integer>>
            implements IterationListener<Tuple2<double[], Integer>> {
        private final List<Factors> factorsList = new ArrayList<>();
        private final int numFactors;
        private final int numUserBlocks;
        private final int numItemBlocks;
        private final double[] blockYty;

        public ComputeLocalYty(int numFactors, final int numUserBlocks, final int numItemBlocks) {
            this.numFactors = numFactors;
            blockYty = new double[numFactors * numFactors];
            this.numUserBlocks = numUserBlocks;
            this.numItemBlocks = numItemBlocks;
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark,
                Context context,
                Collector<Tuple2<double[], Integer>> collector) {

            int tmpStep = epochWatermark % (numUserBlocks + numItemBlocks);
            int tmpIdentity = (tmpStep >= numUserBlocks) ? 0 : 1;
            if (tmpStep == 0 || tmpStep == numUserBlocks) {
                Arrays.fill(blockYty, 0.);
                for (Factors v : factorsList) {
                    if (v.identity != tmpIdentity) {
                        continue;
                    }
                    double[] buff = new double[numFactors];
                    v.getFactorsAsDoubleArray(buff);

                    float[] factors1 = v.factors;
                    for (int i = 0; i < numFactors; i++) {
                        for (int j = 0; j < numFactors; j++) {
                            blockYty[i * numFactors + j] += factors1[i] * factors1[j];
                        }
                    }
                }
            }
            collector.collect(Tuple2.of(blockYty, epochWatermark));
        }

        @Override
        public void onIterationTerminated(
                Context context, Collector<Tuple2<double[], Integer>> collector) {}

        @Override
        public void flatMap(Factors factors, Collector<Tuple2<double[], Integer>> collector)
                throws Exception {
            factorsList.add(factors);
        }
    }

    /** Factors of a user or an item. */
    public static class Factors {
        /**
         * If identity is 0, then it is a user factors. if identity is 1, then it is an item
         * factors.
         */
        public byte identity;

        /* UserId or itemId decided by identity. */
        public long nodeId;

        /* Factors of this nodeId. */
        public float[] factors;

        public Factors() {}

        /**
         * Since this algorithm uses double precision to solve the least square problem, we need to
         * give the converting functions between the factors and the double array.
         */
        public void getFactorsAsDoubleArray(double[] buffer) {
            for (int i = 0; i < factors.length; i++) {
                buffer[i] = factors[i];
            }
        }

        public void copyFactorsFromDoubleArray(double[] buffer) {
            if (factors == null) {
                factors = new float[buffer.length];
            }
            for (int i = 0; i < buffer.length; i++) {
                factors[i] = (float) buffer[i];
            }
        }
    }

    /** The whole ratings of a user or an item. */
    public static class Ratings {

        public Ratings() {}

        /**
         * If identity is 0, then this is a user ratings. if identity is 1, then this is an item
         * ratings.
         */
        public byte identity;

        /* UserId or itemId decided by identity. */
        public long nodeId;

        /* Neighbors of this nodeId. */
        public long[] neighbors;

        /* Ratings from neighbors to this nodeId. */
        public float[] ratings;
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
}
