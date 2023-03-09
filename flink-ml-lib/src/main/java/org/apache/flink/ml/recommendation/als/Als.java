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
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichCoGroupFunction;
import org.apache.flink.api.common.functions.RichFilterFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
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
import org.apache.flink.streaming.api.operators.BoundedOneInput;
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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

/** An Estimator which implements the Als algorithm. */
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
                trainData
                        .map(
                                (MapFunction<Row, Tuple3<Long, Long, Float>>)
                                        value -> {
                                            Number user = value.getFieldAs(userCol);
                                            Number item = value.getFieldAs(itemCol);
                                            Number rating =
                                                    ratingCol == null
                                                            ? 0.0F
                                                            : value.getFieldAs(ratingCol);

                                            return new Tuple3<>(
                                                    user.longValue(),
                                                    item.longValue(),
                                                    rating.floatValue());
                                        })
                        .returns(Types.TUPLE(Types.LONG, Types.LONG, Types.FLOAT));

        DataStream<Ratings> graphData = initGraph(alsInput);
        DataStream<Factors> userItemFactors = initFactors(graphData, getRank(), getSeed());

        DataStream<List<Factors>> result =
                Iterations.iterateBoundedStreamsUntilTermination(
                                DataStreamList.of(userItemFactors),
                                ReplayableDataStreamList.replay(graphData),
                                IterationConfig.newBuilder()
                                        .setOperatorLifeCycle(OperatorLifeCycle.PER_ROUND)
                                        .build(),
                                new TrainIterationBody(
                                        getRank(),
                                        getNonnegative(),
                                        getMaxIter(),
                                        getImplicitprefs(),
                                        getRegParam(),
                                        getAlpha(),
                                        getNumUserBlocks(),
                                        getNumItemBlocks()))
                        .get(0);

        DataStream<AlsModelData> modelData =
                result.transform(
                                "generateModelData",
                                TypeInformation.of(AlsModelData.class),
                                new GenerateModelData())
                        .setParallelism(1);

        AlsModel model = new AlsModel().setModelData(tEnv.fromDataStream(modelData));
        ReadWriteUtils.updateExistingParams(model, paramMap);
        return model;
    }

    private static class GenerateModelData extends AbstractStreamOperator<AlsModelData>
            implements OneInputStreamOperator<List<Factors>, AlsModelData>, BoundedOneInput {
        private final List<Tuple2<Long, float[]>> userFactors = new ArrayList<>();
        private final List<Tuple2<Long, float[]>> itemFactors = new ArrayList<>();

        @Override
        public void endInput() throws Exception {
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
            System.out.println("outputModel OK.");
        }
    }

    private static class TrainIterationBody implements IterationBody {
        private final int numFactors;
        private final boolean nonNegative;
        private final int maxIter;
        private final boolean implicitPrefs;
        private final double regParam;
        private final double alpha;
        private final int numUserBatches;
        private final int numItemBatches;

        public TrainIterationBody(
                int numFactors,
                boolean nonNegative,
                int maxIter,
                boolean implicitPrefs,
                double regParam,
                double alpha,
                int numUserBatches,
                int numItemBatches) {
            this.numFactors = numFactors;
            this.nonNegative = nonNegative;
            this.maxIter = maxIter;
            this.implicitPrefs = implicitPrefs;
            this.regParam = regParam;
            this.alpha = alpha;
            this.numUserBatches = numUserBatches;
            this.numItemBatches = numItemBatches;
        }

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
			DataStream<Factors> userAndItemFactors = variableStreams.get(0);

            DataStreamList feedbackVariableStream =
                    IterationBody.forEachRound(
                            dataStreams,
                            input -> {
                                DataStream<Ratings> graphData = dataStreams.get(0);
                                DataStream<Factors> factors =
                                        updateFactors(
                                                userAndItemFactors,
                                                graphData,
                                                numFactors,
                                                nonNegative,
                                                implicitPrefs,
                                                regParam,
                                                alpha,
                                                numUserBatches,
                                                numItemBatches);
                                return DataStreamList.of(factors);
                            });

            DataStream<Factors> feedbackFactors = variableStreams.get(0);

            final OutputTag<List<Factors>> modelDataOutputTag =
                    new OutputTag<List<Factors>>("MODEL_OUTPUT") {};

            SingleOutputStreamOperator<Integer> iterationController =
                    feedbackFactors.flatMap(
                            new IterationControllerFunc(
                                    modelDataOutputTag, maxIter, numUserBatches, numItemBatches));

            return new IterationBodyResult(
                    feedbackVariableStream,
                    DataStreamList.of(iterationController.getSideOutput(modelDataOutputTag)),
                    iterationController);
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
                                    reusedFactors.factors[i] =
                                            .1F * (i + 1) / 10.0F; // random.nextFloat();
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
                .keyBy(
                        (KeySelector<Tuple4<Long, Long, Float, Byte>, String>)
                                value -> value.f3.toString() + value.f0)
                .window(EndOfStreamWindows.get())
                .process(
                        new ProcessWindowFunction<
                                Tuple4<Long, Long, Float, Byte>, Ratings, String, TimeWindow>() {

                            @Override
                            public void process(
                                    String o,
                                    Context context,
                                    Iterable<Tuple4<Long, Long, Float, Byte>> iterable,
                                    Collector<Ratings> collector) {
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
                        })
                .returns(GenericTypeInfo.of(Ratings.class))
                .name("init_graph");
    }

    private static class StepFunction extends AbstractStreamOperator<Integer>
            implements OneInputStreamOperator<Factors, Integer>,
                    BoundedOneInput,
                    IterationListener<Integer> {

        @Override
        public void endInput() throws Exception {
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
     * @param userAndItemFactors Users' and items' factors at the beginning of the step.
     * @param graphData Users' and items' ratings.
     * @param numFactors Number of factors.
     * @param nonNegative Whether to enforce non-negativity constraint.
     * @return Tuple2 of all factors and stop criterion.
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    private static DataStream<Factors> updateFactors(
            DataStream<Factors> userAndItemFactors,
            DataStream<Ratings> graphData,
            final int numFactors,
            final boolean nonNegative,
            final boolean implicitPrefs,
            final double regParam,
            final double alpha,
            final int numUserBatches,
            final int numItemBatches) {

        SingleOutputStreamOperator<Integer> step =
                userAndItemFactors.transform("step", Types.INT, new StepFunction());

        Map<String, DataStream<?>> broadcastMap = new HashMap<>();
        broadcastMap.put("step", step);

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
                                                private int userOrItem;
                                                private int subStepNo = -1;
                                                private int numSubSteps;

                                                @Override
                                                public boolean filter(Ratings value) {
                                                    if (subStepNo == -1) {
                                                        List<Object> broadStep =
                                                                getRuntimeContext()
                                                                        .getBroadcastVariable(
                                                                                "step");
                                                        int step =
                                                                broadStep.size() > 0
                                                                        ? (int) broadStep.get(0)
                                                                        : -1;
                                                        if (step == -1) {
                                                            subStepNo = -2;
                                                            userOrItem = -1;
                                                        } else {
                                                            int tmpStep =
                                                                    step
                                                                            % (numUserBatches
                                                                                    + numItemBatches);
                                                            if (tmpStep < numUserBatches) {
                                                                subStepNo = tmpStep;
                                                                userOrItem = 0;
                                                            } else {
                                                                subStepNo =
                                                                        tmpStep - numUserBatches;
                                                                userOrItem = 1;
                                                            }
                                                        }

                                                        numSubSteps =
                                                                (userOrItem == 0)
                                                                        ? numUserBatches
                                                                        : numItemBatches;
                                                    }
                                                    return value.identity == userOrItem
                                                            && Math.abs(value.identity)
                                                                            % numSubSteps
                                                                    == subStepNo;
                                                }
                                            });
                                })
                        .map(
                                new RichMapFunction<Ratings, Tuple2<Integer, Ratings>>() {
                                    transient int partitionId;

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

                                    private transient int[] flag = null;
                                    private transient int[] partitionsIds = null;

                                    @Override
                                    public void open(Configuration parameters) {
                                        int numTasks =
                                                getRuntimeContext().getNumberOfParallelSubtasks();
                                        flag = new int[numTasks];
                                        partitionsIds = new int[numTasks];
                                    }

                                    @Override
                                    public void close() {
                                        flag = null;
                                        partitionsIds = null;
                                    }

                                    @Override
                                    public void coGroup(
                                            Iterable<Tuple3<Integer, Byte, Long>> request,
                                            Iterable<Factors> factorsStore,
                                            Collector<Tuple2<Integer, Factors>> out) {

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
                                });

        DataStream<Factors> updatedBatchFactors;

        // Calculate factors
        if (implicitPrefs) {
            DataStream<double[]> yty =
                    computeYtY(userAndItemFactors, numFactors, numUserBatches, numItemBatches);

            // Tuple: Identity, nodeId, factors
            updatedBatchFactors =
                    BroadcastUtils.withBroadcastStream(
                            Arrays.asList(miniBatch, response),
                            Collections.singletonMap("YtY", yty),
                            inputList -> {
                                DataStream<Tuple2<Integer, Ratings>> miniBatchRatings =
                                        (DataStream<Tuple2<Integer, Ratings>>) inputList.get(0);
                                DataStream<Tuple2<Integer, Factors>> responseData =
                                        (DataStream<Tuple2<Integer, Factors>>) inputList.get(1);
                                // Tuple: partitionId, Ratings
                                return miniBatchRatings
                                        .coGroup(responseData) // Tuple: partitionId, Factors
                                        .where(value -> value.f0)
                                        .equalTo(value -> value.f0)
                                        .window(EndOfStreamWindows.get())
                                        .apply(
                                                new UpdateFactorsFunc(
                                                        false,
                                                        numFactors,
                                                        regParam,
                                                        alpha,
                                                        nonNegative));
                            });
        } else {
            // Tuple: Identity, nodeId, factors
            updatedBatchFactors =
                    miniBatch // Tuple: partitionId, Ratings
                            .coGroup(response) // Tuple: partitionId, Factors
                            .where(value -> value.f0)
                            .equalTo(value -> value.f0)
                            .window(EndOfStreamWindows.get())
                            .apply(
                                    new UpdateFactorsFunc(
                                            true,
                                            numFactors,
                                            regParam,
                                            nonNegative)); // .name("CalculateNewFactorsExplicit");
        }

        return userAndItemFactors
                .coGroup(updatedBatchFactors)
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
                            public void coGroup(
                                    Iterable<Factors> old,
                                    Iterable<Factors> updated,
                                    Collector<Factors> out) {

                                assert (old != null);
                                Iterator<Factors> iterator;

                                if (updated == null || !(iterator = updated.iterator()).hasNext()) {
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
    }

    private static class IterationControllerFunc
            implements FlatMapFunction<Factors, Integer>, IterationListener<Integer> {
        private final OutputTag<List<Factors>> modelDataOutputTag;
        private final List<Factors> factorsList = new ArrayList<>();
        private final int maxIter;
        private final int numUserBatches;
        private final int numItemBatches;

        public IterationControllerFunc(
                OutputTag<List<Factors>> modelDataOutputTag,
                int maxIter,
                int numUserBatches,
                int numItemBatches) {
            this.modelDataOutputTag = modelDataOutputTag;
            this.maxIter = maxIter;
            this.numUserBatches = numUserBatches;
            this.numItemBatches = numItemBatches;
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<Integer> collector) {
            if ((epochWatermark + 1) < maxIter * (numUserBatches + numItemBatches)) {
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
        final boolean nonNegative;

        private int numNodes = 0;
        private long numEdges = 0L;
        private long numNeighbors = 0L;
        private boolean firstStep = true;
        private transient double[] yty = null;

        UpdateFactorsFunc(boolean explicit, int numFactors, double lambda, boolean nonNegative) {
            this.explicit = explicit;
            this.numFactors = numFactors;
            this.lambda = lambda;
            this.alpha = 0.;
            this.nonNegative = nonNegative;
        }

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
        public void open(Configuration parameters) {
            numNodes = 0;
            numEdges = 0;
            numNeighbors = 0L;
        }

        @Override
        public void close() {
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
                    yty = (double[]) (getRuntimeContext().getBroadcastVariable("YtY").get(0));
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
                    ls.solve(x, nonNegative);
                } else {
                    ls.merge(new DenseMatrix(numFactors, numFactors, yty));

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
                        cachedFactors.get(pos).f1.getFactorsAsDoubleArray(buffer.values);
                        ls.add(buffer, ((r > 0.0) ? (1.0 + c1) : 0.0), c1);
                    }
                    numExplicit = Math.max(numExplicit, 1);
                    ls.regularize(numExplicit * lambda);
                    ls.solve(x, nonNegative);
                    // System.out.println(x);
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
            DataStream<Factors> factors,
            final int numFactors,
            final int numUserBatches,
            final int numItemBatches) {

        SingleOutputStreamOperator<double[]> localYtY =
                factors.flatMap(new ComputeLocalYtY(numFactors, numUserBatches, numItemBatches));

        return DataStreamUtils.reduce(
                localYtY,
                (ReduceFunction<double[]>)
                        (value1, value2) -> {
                            int n2 = numFactors * numFactors;

                            for (int j = 0; j < n2; ++j) {
                                value1[j] += value2[j];
                            }
                            return value1;
                        });
    }

    private static class ComputeLocalYtY
            implements FlatMapFunction<Factors, double[]>, IterationListener<double[]> {
        private final List<Factors> factorsList = new ArrayList<>();
        private final int numFactors;
        private final int numUserBatches;
        private final int numItemBatches;
        private final double[] blockYtY;

        public ComputeLocalYtY(int numFactors, final int numUserBatches, final int numItemBatches) {
            this.numFactors = numFactors;
            blockYtY = new double[numFactors * numFactors];
            this.numUserBatches = numUserBatches;
            this.numItemBatches = numItemBatches;
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<double[]> collector) {

            int tmpStep = epochWatermark % (numUserBatches + numItemBatches);
            int tmpIdentity = (tmpStep >= numUserBatches) ? 0 : 1;

            Arrays.fill(blockYtY, 0.);

            for (Factors v : factorsList) {
                if (v.identity != tmpIdentity) {
                    continue;
                }

                float[] factors1 = v.factors;
                for (int i = 0; i < numFactors; i++) {
                    for (int j = 0; j < numFactors; j++) {
                        blockYtY[i * numFactors + j] += factors1[i] * factors1[j];
                    }
                }
            }
            System.out.println("computeYtY OK." + epochWatermark);

            collector.collect(blockYtY);
        }

        @Override
        public void onIterationTerminated(Context context, Collector<double[]> collector) {}

        @Override
        public void flatMap(Factors factors, Collector<double[]> collector) throws Exception {
            factorsList.add(factors);
        }
    }

	/** Factors of a user or an item. */
    public static class Factors {
		/**
		 * If identity is 0, then this is a user factors.
		 * if identity is 1, then this is an item factors.
		 */
		public byte identity;
		/**
		 * UserId or itemId decided by identity.
		 */
		public long nodeId;
        public float[] factors;

        public Factors() {}

        /**
         * Since this algorithm uses double precision to solve the least square problem, we need to convert the
         * factors to double array.
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

    /** All ratings of a user or an item. */
    public static class Ratings {
        public Ratings() {}

		/**
		 * If identity is 0, then this is a user ratings.
		 * if identity is 1, then this is an item ratings.
		 */
        public byte identity;
		/**
		 * UserId or itemId decided by identity.
		 */
        public long nodeId;
        public long[] neighbors;
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
