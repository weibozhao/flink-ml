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

package org.apache.flink.ml.common.ps;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.ps.api.ServerIteration;
import org.apache.flink.ml.common.ps.iterations.BaseComponent;
import org.apache.flink.ml.common.ps.iterations.MLSession;
import org.apache.flink.ml.common.ps.iterations.ProcessComponent;
import org.apache.flink.ml.common.ps.iterations.PsAllReduceComponent;
import org.apache.flink.ml.common.ps.iterations.PullComponent;
import org.apache.flink.ml.common.ps.iterations.PushComponent;
import org.apache.flink.ml.common.ps.iterations.ReduceScatterComponent;
import org.apache.flink.ml.common.ps.sarray.SharedDoubleArray;
import org.apache.flink.ml.common.ps.sarray.SharedFloatArray;
import org.apache.flink.ml.common.ps.sarray.SharedLongArray;
import org.apache.flink.ml.common.ps.utils.ProxySideOutput;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.runtime.util.ResettableIterator;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;

import java.util.Iterator;
import java.util.function.Function;

/**
 * The worker operator that executes the iterative machine learning process following {@link
 * ServerIteration}.
 *
 * <p>In detail, the worker operator is responsible for the following:
 *
 * <ul>
 *   <li>Caches the training data.
 *   <li>Initializes the {@link MLSession}.
 *   <li>Splits the {@link ServerIteration} by {@link PullComponent}, {@link PsAllReduceComponent}
 *       and {@link ReduceScatterComponent} into multiple sequences and map it into
 *       flink-ml-iterations.
 *   <li>Executes the process function in each {@link ProcessComponent}.
 *   <li>Executes the push/pull/all-reduce/reduce-scatter request in {@link PushComponent}, {@link
 *       PullComponent}, {@link PsAllReduceComponent} and {@link ReduceScatterComponent}. which
 *       talks to servers, by reading/writing {@link MLSession}.
 * </ul>
 */
public class WorkerOperator<DT, SessionT extends MLSession> extends AbstractStreamOperator<byte[]>
        implements TwoInputStreamOperator<DT, byte[], byte[]>, IterationListener<byte[]> {
    /** The user defined iteration logic. */
    private final ServerIteration<SessionT> iterationStages;
    /**
     * Iteration id in terms of {@link ServerIteration}. When we finished processing all stages in
     * stageList, the iteration id increments by one.
     */
    private int iterationId;

    /** The id of the stages to execute in iterationStages. */
    private int nextStageToExecute = 0;

    private ListState<Integer> nextStageToExecuteState;

    /** The agent for each worker to talk with servers. */
    private transient ServerAgent serverAgent;
    /** Number of servers that this worker needs to talk to. */
    private final int numServers;
    /** The hash function to distribute keys to servers. */
    private transient Function<Long, Integer> hashFunc;

    /** The cached training data. */
    private ListStateWithCache<DT> trainDataState;

    /**
     * Number of segments received from servers for the current request. For each request, a worker
     * should receive one segment from each server.
     */
    private int numSegmentsReceived = 0;

    private ListState<Integer> numSegmentsReceivedState;

    /**
     * The memory store for pull answer. For a pull request, each received segment will be filled to
     * the user provided buffer.
     */
    private double[] pulledResult;

    private float[] pulledFloatResult;
    private ListState<double[]> pulledResultState;
    private ListState<float[]> pulledFloatResultState;

    /** The state store for the all-reduce/reduce-scatter results. */
    private ListState<byte[]> reducedResult;

    public WorkerOperator(ServerIteration<SessionT> iterationStages, int numServers) {
        this.iterationStages = iterationStages;
        this.numServers = numServers;
    }

    @Override
    public void open() {
        int workerId = getRuntimeContext().getIndexOfThisSubtask();
        int numWorkers = getRuntimeContext().getNumberOfParallelSubtasks();
        this.hashFunc = key -> (int) (Math.abs(key % numServers));
        this.serverAgent = new ServerAgent(workerId, numServers, hashFunc, output);
        iterationStages.session.setWorldInfo(workerId, numWorkers);
        iterationStages.session.setOutput(new ProxySideOutput(output));
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<byte[]> collector) throws Exception {
        if (epochWatermark == 0) {
            iterationStages.session.setInputData(new ResettableTrainDataIterator<>(trainDataState));
            nextStageToExecute = processIterationStages(nextStageToExecute, iterationStages);
        }
    }

    @Override
    public void onIterationTerminated(Context context, Collector<byte[]> collector) {
        trainDataState.clear();
    }

    @Override
    public void processElement1(StreamRecord<DT> streamRecord) throws Exception {
        trainDataState.add(streamRecord.getValue());
    }

    @Override
    public void processElement2(StreamRecord<byte[]> streamRecord) throws Exception {
        Message message = new Message(streamRecord.getValue());
        BaseComponent stage =
                iterationStages.iterationStageList.get(
                        nextStageToExecute % iterationStages.iterationStageList.size());

        boolean proceedToNextStage;
        if (stage instanceof PullComponent) {
            if (((PullComponent) stage).values.get() instanceof SharedDoubleArray) {
                proceedToNextStage = onPullResponse(message, (PullComponent) stage);
            } else {
                proceedToNextStage = onPullFloatResponse(message, (PullComponent) stage);
            }
        } else if (stage instanceof PsAllReduceComponent) {
            proceedToNextStage = onAllReduceResponse(message, (PsAllReduceComponent<?>) stage);
        } else if (stage instanceof ReduceScatterComponent) {
            proceedToNextStage =
                    onReduceScatterResponse(message, (ReduceScatterComponent<?>) stage);
        } else {
            throw new IllegalStateException(
                    "Illegal stage type: %s" + stage.getClass().getSimpleName() + ".");
        }

        if (proceedToNextStage) {
            nextStageToExecute++;
            nextStageToExecute = processIterationStages(nextStageToExecute, iterationStages);
        }
    }

    private boolean onPullResponse(Message message, PullComponent pullStage) {
        numSegmentsReceived++;
        double[] segment = message.getValuesInDoubleArray();
        if (segment.length != 0) {
            if (pullStage.aggregator != null) {
                if (pulledResult.length == 0) {
                    pulledResult = segment;
                } else {
                    pulledResult = pullStage.aggregator.merge(segment, pulledResult);
                }
            } else {
                SharedLongArray keys = pullStage.keys.get();
                SharedDoubleArray values = (SharedDoubleArray) pullStage.values.get();
                int serverId = message.getServerId();
                long[] keysArray = keys.elements();

                if (pulledResult.length == 0) {
                    pulledResult = values.elements();
                }

                int numDoublesPerKey = values.size() / keys.size();
                // Copy the response from one server to the result array.
                int idxInLocalPull = 0;
                for (int i = 0; i < keys.size(); i++) {
                    if (hashFunc.apply(keysArray[i]) == serverId) {
                        System.arraycopy(
                                segment,
                                idxInLocalPull * numDoublesPerKey,
                                pulledResult,
                                i * numDoublesPerKey,
                                numDoublesPerKey);
                        idxInLocalPull++;
                    }
                }
            }
        }

        if (numSegmentsReceived == numServers) {
            SharedDoubleArray pullPlaceHolder = (SharedDoubleArray) pullStage.values.get();
            System.arraycopy(
                    pulledResult, 0, pullPlaceHolder.elements(), 0, pullPlaceHolder.size());

            pulledResult = new double[0];
            numSegmentsReceived = 0;
            return true;
        }
        return false;
    }

    private boolean onPullFloatResponse(Message message, PullComponent pullStage) {
        numSegmentsReceived++;
        float[] segment = message.getValuesInFLoatArray();
        if (segment.length != 0) {
            if (pullStage.floatAggregator != null) {
                if (pulledFloatResult.length == 0) {
                    pulledFloatResult = segment;
                } else {
                    pulledFloatResult = pullStage.floatAggregator.merge(segment, pulledFloatResult);
                }
            } else {
                SharedLongArray keys = pullStage.keys.get();
                SharedFloatArray values = (SharedFloatArray) pullStage.values.get();
                int serverId = message.getServerId();
                long[] keysArray = keys.elements();

                if (pulledFloatResult.length == 0) {
                    pulledFloatResult = values.elements();
                }

                int numFloatsPerKey = values.size() / keys.size();
                // Copy the response from one server to the result array.
                int idxInLocalPull = 0;
                for (int i = 0; i < keys.size(); i++) {
                    if (hashFunc.apply(keysArray[i]) == serverId) {
                        System.arraycopy(
                                segment,
                                idxInLocalPull * numFloatsPerKey,
                                pulledFloatResult,
                                i * numFloatsPerKey,
                                numFloatsPerKey);
                        idxInLocalPull++;
                    }
                }
            }
        }

        if (numSegmentsReceived == numServers) {
            SharedFloatArray pullPlaceHolder = (SharedFloatArray) pullStage.values.get();
            System.arraycopy(
                    pulledFloatResult, 0, pullPlaceHolder.elements(), 0, pullPlaceHolder.size());

            pulledFloatResult = new float[0];
            numSegmentsReceived = 0;
            return true;
        }
        return false;
    }

    private <V> boolean onAllReduceResponse(Message message, PsAllReduceComponent<V> allReduceStage)
            throws Exception {
        numSegmentsReceived++;
        reducedResult.add(message.bytes);

        if (numSegmentsReceived == numServers) {
            Message assembled = Message.assembleMessages(reducedResult.get().iterator());
            V[] reduceResult = assembled.getValues(allReduceStage.typeSerializer);
            System.arraycopy(reduceResult, 0, allReduceStage.recvBuf.get(), 0, reduceResult.length);
            reducedResult.clear();
            numSegmentsReceived = 0;
            return true;
        }
        return false;
    }

    private <V> boolean onReduceScatterResponse(
            Message message, ReduceScatterComponent<V> reduceScatterStage) throws Exception {
        numSegmentsReceived++;
        reducedResult.add(message.bytes);

        if (numSegmentsReceived == numServers) {
            Message assembled = Message.assembleMessages(reducedResult.get().iterator());
            V[] reduceResult = assembled.getValues(reduceScatterStage.typeSerializer);
            System.arraycopy(
                    reduceResult, 0, reduceScatterStage.recvBuf.get(), 0, reduceResult.length);
            reducedResult.clear();
            numSegmentsReceived = 0;
            return true;
        }
        return false;
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        trainDataState =
                new ListStateWithCache<>(
                        (getOperatorConfig().getTypeSerializerIn(0, getClass().getClassLoader())),
                        getContainingTask(),
                        getRuntimeContext(),
                        context,
                        config.getOperatorID());

        numSegmentsReceivedState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>("numSegmentsReceivedState", Types.INT));
        numSegmentsReceived =
                OperatorStateUtils.getUniqueElement(
                                numSegmentsReceivedState, "numSegmentsReceivedState")
                        .orElse(0);

        nextStageToExecuteState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>("nextStageToExecuteState", Types.INT));

        nextStageToExecute =
                OperatorStateUtils.getUniqueElement(
                                nextStageToExecuteState, "nextStageToExecuteState")
                        .orElse(0);

        iterationStages.session.initializeState(context);

        pulledResultState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "pulledResultState",
                                        PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO));
        pulledResult =
                OperatorStateUtils.getUniqueElement(pulledResultState, "pulledResultState")
                        .orElse(new double[0]);
        pulledFloatResultState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "pulledFloatResultState",
                                        PrimitiveArrayTypeInfo.FLOAT_PRIMITIVE_ARRAY_TYPE_INFO));
        pulledFloatResult =
                OperatorStateUtils.getUniqueElement(
                                pulledFloatResultState, "pulledFloatResultState")
                        .orElse(new float[0]);
        reducedResult =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "reducedResult",
                                        PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO));
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);

        numSegmentsReceivedState.clear();
        numSegmentsReceivedState.add(numSegmentsReceived);

        nextStageToExecuteState.clear();
        nextStageToExecuteState.add(nextStageToExecute);

        trainDataState.snapshotState(context);
        iterationStages.session.snapshotState(context);

        pulledResultState.clear();
        pulledResultState.add(pulledResult);
        pulledFloatResultState.clear();
        pulledFloatResultState.add(pulledFloatResult);
    }

    /**
     * Processes the stages described in the given iterationStages from the given nextStage id. This
     * function processes the stages until it meets a {@link PullComponent}, {@link
     * PsAllReduceComponent} or {@link ReduceScatterComponent}.
     *
     * @param nextStageToExecute id of the next stage to execute in the given iteration stages.
     * @param iterationStages iteration stages used to describe the training logic.
     * @return the id of the next stage to execute.
     */
    @SuppressWarnings("unchecked")
    private <V> int processIterationStages(
            int nextStageToExecute, ServerIteration<SessionT> iterationStages) throws Exception {
        while (true) {
            if (nextStageToExecute > 0
                    && nextStageToExecute % iterationStages.iterationStageList.size() == 0) {
                iterationId = nextStageToExecute / iterationStages.iterationStageList.size();
                iterationStages.session.setIterationId(iterationId);
                if (iterationStages.shouldTerminate.apply(iterationStages.session)) {
                    return -1;
                }
            }
            BaseComponent stage =
                    iterationStages.iterationStageList.get(
                            nextStageToExecute % iterationStages.iterationStageList.size());

            // We are not incrementing nextStageToExecute for
            // PullStage/AllReduceStage/ReduceScatterStage, since we
            // need to wait for response from servers.
            if (stage instanceof PullComponent) {
                PullComponent pullStage = ((PullComponent) stage);
                if (pullStage.values.get() instanceof SharedDoubleArray) {
                    serverAgent.pull(pullStage.keys.get(), nextStageToExecute);
                } else {
                    serverAgent.pullFloat(pullStage.keys.get(), nextStageToExecute);
                }
                return nextStageToExecute;
            } else if (stage instanceof PsAllReduceComponent) {
                PsAllReduceComponent<V> allReduceStage = (PsAllReduceComponent<V>) stage;
                if (iterationId % allReduceStage.executionInterval == 0) {
                    serverAgent.reduce(
                            allReduceStage.sendBuf.get(),
                            allReduceStage.typeSerializer,
                            nextStageToExecute);
                    return nextStageToExecute;
                } else {
                    nextStageToExecute++;
                }

            } else if (stage instanceof ReduceScatterComponent) {
                ReduceScatterComponent<V> reduceScatterStage = (ReduceScatterComponent<V>) stage;
                if (iterationId % reduceScatterStage.executionInterval == 0) {
                    serverAgent.reduce(
                            reduceScatterStage.sendBuf.get(),
                            reduceScatterStage.typeSerializer,
                            nextStageToExecute);
                    return nextStageToExecute;
                } else {
                    nextStageToExecute++;
                }
            } else if (stage instanceof PushComponent) {
                PushComponent pushStage = (PushComponent) stage;
                if (pushStage.values.get() instanceof SharedDoubleArray) {
                    serverAgent.push(
                            pushStage.keys.get(),
                            (SharedDoubleArray) pushStage.values.get(),
                            nextStageToExecute);
                } else {
                    serverAgent.pushFloat(
                            pushStage.keys.get(),
                            (SharedFloatArray) pushStage.values.get(),
                            nextStageToExecute);
                }
                nextStageToExecute++;
            } else if (stage instanceof ProcessComponent) {
                ((ProcessComponent<SessionT>) stage).process(iterationStages.session);
                nextStageToExecute++;
            } else {
                throw new IllegalStateException(
                        "Illegal type of IterationStage: + "
                                + stage.getClass().getSimpleName()
                                + ".");
            }
        }
    }

    /** A resettable iterator for {@link ListStateWithCache}. */
    private static class ResettableTrainDataIterator<T> implements ResettableIterator<T> {
        private final ListStateWithCache<T> data;
        private Iterator<T> dataIterator;

        public ResettableTrainDataIterator(ListStateWithCache<T> data) throws Exception {
            this.data = data;
            this.dataIterator = data.get().iterator();
        }

        @Override
        public void reset() {
            try {
                this.dataIterator = data.get().iterator();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public boolean hasNext() {
            return dataIterator.hasNext();
        }

        @Override
        public T next() {
            return dataIterator.next();
        }
    }
}
