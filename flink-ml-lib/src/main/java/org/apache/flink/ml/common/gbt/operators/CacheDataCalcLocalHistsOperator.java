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

package org.apache.flink.ml.common.gbt.operators;

import org.apache.flink.api.java.tuple.Tuple5;
import org.apache.flink.api.java.typeutils.runtime.kryo.KryoSerializer;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.BoostingStrategy;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.gbt.typeinfo.BinnedInstanceSerializer;
import org.apache.flink.ml.common.lossfunc.LossFunc;
import org.apache.flink.ml.common.sharedobjects.AbstractSharedObjectsStreamOperator;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.concurrent.BasicThreadFactory;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Calculates local histograms for local data partition.
 *
 * <p>This operator only has input elements in the first round, including data instances and raw
 * training context. There will be no input elements in other rounds. The output elements are tuples
 * of (subtask index, (nodeId, featureId) pair index, sliceId, numSlices, Histogram).
 *
 * <p>As there could be many instances in one node, histogram calculation could be expensive. To
 * alleviate the computation costs, multi-threading in introduced where the instances in one node is
 * partitioned into multiple slices, and their statistics are calculated in parallel and sent to
 * {@link ReduceHistogramFunction} for reducing.
 */
public class CacheDataCalcLocalHistsOperator
        extends AbstractSharedObjectsStreamOperator<
                Tuple5<Integer, Integer, Integer, Integer, Histogram>>
        implements TwoInputStreamOperator<
                        Row, TrainContext, Tuple5<Integer, Integer, Integer, Integer, Histogram>>,
                IterationListener<Tuple5<Integer, Integer, Integer, Integer, Histogram>> {

    private static final String TREE_INITIALIZER_STATE_NAME = "tree_initializer";
    private static final String HIST_BUILDER_STATE_NAME = "hist_builder";

    private final BoostingStrategy strategy;
    private final boolean useMultiThreading;

    // States of local data.
    private transient ListStateWithCache<BinnedInstance> instancesCollecting;
    private transient ListStateWithCache<TreeInitializer> treeInitializerState;
    private transient TreeInitializer treeInitializer;
    private transient ListStateWithCache<HistBuilder> histBuilderState;
    private transient HistBuilder histBuilder;
    private transient ExecutorService executor;

    public CacheDataCalcLocalHistsOperator(BoostingStrategy strategy) {
        super();
        this.strategy = strategy;
        useMultiThreading = strategy.numThreads > 1;
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);

        instancesCollecting =
                new ListStateWithCache<>(
                        BinnedInstanceSerializer.INSTANCE,
                        getContainingTask(),
                        getRuntimeContext(),
                        context,
                        getOperatorID());
        treeInitializerState =
                new ListStateWithCache<>(
                        new KryoSerializer<>(TreeInitializer.class, getExecutionConfig()),
                        getContainingTask(),
                        getRuntimeContext(),
                        context,
                        getOperatorID());
        treeInitializer =
                OperatorStateUtils.getUniqueElement(
                                treeInitializerState, TREE_INITIALIZER_STATE_NAME)
                        .orElse(null);
        histBuilderState =
                new ListStateWithCache<>(
                        new KryoSerializer<>(HistBuilder.class, getExecutionConfig()),
                        getContainingTask(),
                        getRuntimeContext(),
                        context,
                        getOperatorID());
        histBuilder =
                OperatorStateUtils.getUniqueElement(histBuilderState, HIST_BUILDER_STATE_NAME)
                        .orElse(null);
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        instancesCollecting.snapshotState(context);
        treeInitializerState.snapshotState(context);
        histBuilderState.snapshotState(context);
    }

    @Override
    public void processElement1(StreamRecord<Row> streamRecord) throws Exception {
        Row row = streamRecord.getValue();
        BinnedInstance instance = new BinnedInstance();
        instance.weight =
                null != strategy.weightCol
                        ? row.<Number>getFieldAs(strategy.weightCol).doubleValue()
                        : 1.;
        instance.label = row.<Number>getFieldAs(strategy.labelCol).doubleValue();

        if (strategy.isInputVector) {
            Vector vec = row.getFieldAs(strategy.featuresCols[0]);
            if (vec instanceof DenseVector) {
                DenseVector dv = (DenseVector) vec;
                instance.featureValues = Arrays.stream(dv.values).mapToInt(d -> (int) d).toArray();
            } else {
                SparseVector sv = (SparseVector) vec;
                instance.featureIds = sv.indices.length == sv.size() ? null : sv.indices;
                instance.featureValues = Arrays.stream(sv.values).mapToInt(d -> (int) d).toArray();
            }
        } else {
            instance.featureValues =
                    Arrays.stream(strategy.featuresCols)
                            .mapToInt(col -> ((Number) row.getFieldAs(col)).intValue())
                            .toArray();
        }
        instancesCollecting.add(instance);
    }

    @Override
    public void processElement2(StreamRecord<TrainContext> streamRecord) throws Exception {
        TrainContext rawTrainContext = streamRecord.getValue();
        invoke(
                (getter, setter) ->
                        setter.set(SharedObjectsConstants.TRAIN_CONTEXT, rawTrainContext));
    }

    public void onEpochWatermarkIncremented(
            int epochWatermark,
            Context context,
            Collector<Tuple5<Integer, Integer, Integer, Integer, Histogram>> out)
            throws Exception {
        if (0 == epochWatermark) {
            // Initializes local state in first round.
            invoke(
                    (getter, setter) -> {
                        BinnedInstance[] instances =
                                (BinnedInstance[])
                                        IteratorUtils.toArray(
                                                instancesCollecting.get().iterator(),
                                                BinnedInstance.class);
                        setter.set(SharedObjectsConstants.INSTANCES, instances);
                        instancesCollecting.clear();

                        TrainContext rawTrainContext =
                                getter.get(SharedObjectsConstants.TRAIN_CONTEXT);
                        TrainContext trainContext =
                                new TrainContextInitializer(strategy)
                                        .init(
                                                rawTrainContext,
                                                getRuntimeContext().getIndexOfThisSubtask(),
                                                getRuntimeContext().getNumberOfParallelSubtasks(),
                                                instances);
                        setter.set(SharedObjectsConstants.TRAIN_CONTEXT, trainContext);

                        treeInitializer = new TreeInitializer(trainContext);
                        treeInitializerState.update(Collections.singletonList(treeInitializer));
                        histBuilder = new HistBuilder(trainContext);
                        histBuilderState.update(Collections.singletonList(histBuilder));
                    });
        }
        if (useMultiThreading && null == executor) {
            BasicThreadFactory factory =
                    new BasicThreadFactory.Builder()
                            .namingPattern(
                                    "gbt-hist-builder-"
                                            + getRuntimeContext().getIndexOfThisSubtask()
                                            + "-thread-%d")
                            .build();
            final int numThreads = strategy.numThreads;
            LOG.info("Use {} threads", numThreads);
            executor = Executors.newFixedThreadPool(numThreads + 1, factory);
        }

        invoke(
                (getter, setter) -> {
                    TrainContext trainContext = getter.get(SharedObjectsConstants.TRAIN_CONTEXT);
                    Preconditions.checkArgument(
                            getRuntimeContext().getIndexOfThisSubtask() == trainContext.subtaskId);
                    BinnedInstance[] instances = getter.get(SharedObjectsConstants.INSTANCES);
                    double[] pgh = getter.get(SharedObjectsConstants.PREDS_GRADS_HESSIANS);
                    // In the first round, use prior as the predictions.
                    if (0 == pgh.length) {
                        pgh = new double[instances.length * 3];
                        double prior = trainContext.prior;
                        LossFunc loss = trainContext.loss;
                        for (int i = 0; i < instances.length; i += 1) {
                            double label = instances[i].label;
                            double weight = instances[i].weight;
                            pgh[3 * i] = prior;
                            pgh[3 * i + 1] = loss.gradient(prior, label) * weight;
                            pgh[3 * i + 2] = loss.hessian(prior, label) * weight;
                        }
                    }

                    boolean needInitTree = getter.get(SharedObjectsConstants.NEED_INIT_TREE);
                    int[] indices;
                    List<LearningNode> layer;
                    if (needInitTree) {
                        // When last tree is finished, initializes a new tree, and shuffle instance
                        // indices.
                        treeInitializer.init(
                                getter.get(SharedObjectsConstants.ALL_TREES).size(),
                                d -> setter.set(SharedObjectsConstants.SHUFFLED_INDICES, d));
                        LearningNode rootLearningNode = treeInitializer.getRootLearningNode();
                        indices = getter.get(SharedObjectsConstants.SHUFFLED_INDICES);
                        layer = Collections.singletonList(rootLearningNode);
                        setter.set(SharedObjectsConstants.ROOT_LEARNING_NODE, rootLearningNode);
                        setter.set(SharedObjectsConstants.HAS_INITED_TREE, true);
                    } else {
                        // Otherwise, uses the swapped instance indices.
                        indices = getter.get(SharedObjectsConstants.SWAPPED_INDICES);
                        layer = getter.get(SharedObjectsConstants.LAYER);
                        setter.set(SharedObjectsConstants.SHUFFLED_INDICES, new int[0]);
                        setter.set(SharedObjectsConstants.HAS_INITED_TREE, false);
                    }

                    if (useMultiThreading) {
                        OutputQueuedCollector collector = new OutputQueuedCollector(executor, out);
                        histBuilder.build(
                                layer,
                                indices,
                                instances,
                                pgh,
                                d -> setter.set(SharedObjectsConstants.NODE_FEATURE_PAIRS, d),
                                executor,
                                strategy.numThreads,
                                collector);
                    } else {
                        histBuilder.build(
                                layer,
                                indices,
                                instances,
                                pgh,
                                d -> setter.set(SharedObjectsConstants.NODE_FEATURE_PAIRS, d),
                                out);
                    }
                });
    }

    @Override
    public void onIterationTerminated(
            Context context,
            Collector<Tuple5<Integer, Integer, Integer, Integer, Histogram>> collector)
            throws Exception {
        instancesCollecting.clear();
        treeInitializerState.clear();
        histBuilderState.clear();

        invoke(
                (getter, setter) -> {
                    setter.set(SharedObjectsConstants.INSTANCES, new BinnedInstance[0]);
                    setter.set(SharedObjectsConstants.SHUFFLED_INDICES, new int[0]);
                    setter.set(SharedObjectsConstants.NODE_FEATURE_PAIRS, new int[0]);
                });
    }

    @Override
    public void close() throws Exception {
        if (useMultiThreading) {
            executor.shutdown();
            // Wait long enough for all threads are terminated.
            Preconditions.checkState(executor.awaitTermination(1, TimeUnit.MINUTES));
        }
        instancesCollecting.clear();
        treeInitializerState.clear();
        histBuilderState.clear();
        super.close();
    }

    static class OutputQueuedCollector
            implements Collector<Tuple5<Integer, Integer, Integer, Integer, Histogram>> {
        private static final Tuple5<Integer, Integer, Integer, Integer, Histogram> END_MARKER =
                Tuple5.of(null, null, null, null, null);
        private final ExecutorService executorService;
        private final Collector<Tuple5<Integer, Integer, Integer, Integer, Histogram>> out;
        private BlockingQueue<Tuple5<Integer, Integer, Integer, Integer, Histogram>> outputQueue;
        private Future<Void> future;

        private OutputQueuedCollector(
                ExecutorService executorService,
                Collector<Tuple5<Integer, Integer, Integer, Integer, Histogram>> out) {
            this.executorService = executorService;
            this.out = out;
        }

        private static boolean isEndMarker(
                Tuple5<Integer, Integer, Integer, Integer, Histogram> e) {
            return null == e.f0;
        }

        private Void run() throws InterruptedException {
            Tuple5<Integer, Integer, Integer, Integer, Histogram> e;
            while (true) {
                e = outputQueue.take();
                if (isEndMarker(e)) {
                    break;
                }
                out.collect(e);
            }
            return null;
        }

        public void start() {
            outputQueue = new LinkedBlockingQueue<>();
            future = executorService.submit(this::run);
        }

        public void awaitTermination() throws ExecutionException, InterruptedException {
            outputQueue.put(END_MARKER);
            future.get();
        }

        @Override
        public void collect(Tuple5<Integer, Integer, Integer, Integer, Histogram> record) {
            outputQueue.add(record);
        }

        @Override
        public void close() {}
    }
}
