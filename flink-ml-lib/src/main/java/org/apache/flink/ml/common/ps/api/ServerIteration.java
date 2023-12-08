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

package org.apache.flink.ml.common.ps.api;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.Partitioner;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.common.ps.Message;
import org.apache.flink.ml.common.ps.ServerOperator;
import org.apache.flink.ml.common.ps.WorkerOperator;
import org.apache.flink.ml.common.ps.iterations.BaseComponent;
import org.apache.flink.ml.common.ps.iterations.CommonComponent;
import org.apache.flink.ml.common.ps.iterations.MLSession;
import org.apache.flink.ml.common.ps.iterations.PsAllReduceComponent;
import org.apache.flink.ml.common.ps.iterations.PullComponent;
import org.apache.flink.ml.common.ps.iterations.ReduceScatterComponent;
import org.apache.flink.ml.common.ps.updater.ModelUpdater;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;
import org.apache.flink.util.function.SerializableFunction;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * A list of iteration stages to express the logic of an iterative machine learning process.
 *
 * <p>Note that there should be at least one stage (e.g., {@link PullComponent}, {@link
 * PsAllReduceComponent} or {@link ReduceScatterComponent}) that needs to wait for responses from
 * servers.
 */
public class ServerIteration<T extends MLSession> implements CommonComponent, Serializable {
    /** The session on each worker. */
    public T session;
    /** The termination criteria. */
    public Function<T, Boolean> shouldTerminate;

    /** The stage list that describes the iterative process. */
    public List<BaseComponent> iterationStageList;

    public ModelUpdater modelUpdater;

    public ServerIteration(T session) {
        this.session = session;
        iterationStageList = new ArrayList<>();
    }

    /** Sets the criteria of termination. */
    public void stopIteration(SerializableFunction<T, Boolean> shouldTerminate) {
        boolean waitServer = false;
        for (BaseComponent stage : iterationStageList) {
            if (stage instanceof PullComponent
                    || stage instanceof PsAllReduceComponent
                    || stage instanceof ReduceScatterComponent) {
                waitServer = true;
                break;
            }
        }
        Preconditions.checkState(
                waitServer,
                String.format(
                        "There should be at least one stage that needs to receive response from servers (i.e., %s, %s, %s).\n",
                        PullComponent.class.getSimpleName(),
                        PsAllReduceComponent.class.getSimpleName(),
                        ReduceScatterComponent.class.getSimpleName()));
        this.shouldTerminate = shouldTerminate;
    }

    /** Adds a stage into the stage list. */
    public ServerIteration<T> add(BaseComponent stage) {
        iterationStageList.add(stage);
        return this;
    }

    @Override
    public MLData apply(MLData inputData) {
        int numServers = Math.max(1, inputData.getParallelism() / 2);
        if (this.iterationStageList.size() != 0) {
            DataStream<byte[]> variableStream =
                    inputData
                            .getExecutionEnvironment()
                            .fromElements(new byte[0])
                            .filter(x -> false);
            // List<MLData> sideOutputs = inputData.getSideOutputs();
            inputData.add(
                    "psResult",
                    Iterations.iterateBoundedStreamsUntilTermination(
                                    DataStreamList.of(variableStream),
                                    ReplayableDataStreamList.notReplay(
                                            inputData.getCurrentDataStream()),
                                    IterationConfig.newBuilder().build(),
                                    new TrainIterationBody(modelUpdater, this, numServers))
                            .get(0));
        }
        return inputData.slice("psResult");
    }

    /** The iteration implementation for training process. */
    private static class TrainIterationBody implements IterationBody {
        private final ModelUpdater modelUpdater;
        private final ServerIteration<? extends MLSession> serverIteration;
        private final int numServers;

        public TrainIterationBody(
                ModelUpdater modelUpdater,
                ServerIteration<? extends MLSession> serverIteration,
                int numServers) {
            this.serverIteration = serverIteration;
            this.modelUpdater = modelUpdater;
            this.numServers = numServers;
        }

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<byte[]> variableStream = variableStreams.get(0);
            DataStream<LabeledPointWithWeight> trainData = dataStreams.get(0);
            final OutputTag<Object> modelDataOutputTag =
                    new OutputTag<>("MODEL_OUTPUT", TypeInformation.of(Object.class));

            SingleOutputStreamOperator<byte[]> messageToServer =
                    trainData
                            .connect(variableStream)
                            .transform(
                                    "WorkerOp",
                                    PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO,
                                    new WorkerOperator<>(serverIteration, numServers));
            int numWorkers = messageToServer.getParallelism();

            SingleOutputStreamOperator<byte[]> messageToWorker =
                    messageToServer
                            .partitionCustom(
                                    (Partitioner<Integer>)
                                            (key, numPartitions) -> key % numPartitions,
                                    (KeySelector<byte[], Integer>)
                                            value -> new Message(value).getServerId())
                            .transform(
                                    "ServerOp",
                                    PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO,
                                    new ServerOperator(
                                            serverIteration.iterationStageList,
                                            numWorkers,
                                            modelUpdater,
                                            modelDataOutputTag));
            messageToWorker.setParallelism(numServers);

            DataStream<byte[]> feedback =
                    messageToWorker
                            .partitionCustom(
                                    (Partitioner<Integer>)
                                            (key, numPartitions) -> key % numPartitions,
                                    (KeySelector<byte[], Integer>)
                                            value -> new Message(value).getWorkerId())
                            .map(
                                    (MapFunction<byte[], byte[]>) message -> message,
                                    PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO)
                            .setParallelism(numWorkers);

            DataStream<Object> model = messageToWorker.getSideOutput(modelDataOutputTag);

            List<DataStream<?>> result = new ArrayList<>();
            result.add(model);

            List<OutputTag<?>> sideOutputTags = serverIteration.session.getOutputTags();
            if (sideOutputTags != null) {
                for (OutputTag<?> outputTag : sideOutputTags) {
                    result.add(messageToServer.getSideOutput(outputTag));
                }
            }

            return new IterationBodyResult(
                    DataStreamList.of(feedback), new DataStreamList(result), null);
        }
    }
}
