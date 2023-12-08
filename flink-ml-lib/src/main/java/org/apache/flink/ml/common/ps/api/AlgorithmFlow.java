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

import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.common.ps.iterations.BaseComponent;
import org.apache.flink.ml.common.ps.iterations.CommonComponent;
import org.apache.flink.ml.common.ps.iterations.MLSession;
import org.apache.flink.ml.common.ps.updater.ModelUpdater;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.function.SerializableFunction;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/** A list of stages to express the logic of an machine learning algorithm. */
public class AlgorithmFlow implements CommonComponent, Serializable {

    /** The stage list that describes the process. */
    public List<BaseComponent> stages;

    public List<BaseComponent> iterationStageList;
    public ServerIteration<?> serverIteration;

    public List<BaseComponent> beforeIterationStageList;
    public List<BaseComponent> afterIterationStageList;

    public boolean isBeforeIterationStage = true;
    public boolean isInIterationStage = false;
    public boolean isInPsIterationStage = false;

    private String[] inputDataNames;
    private String[] feedbackDataNames;

    private String[] outputNames;

    private boolean isReplayData;

    private boolean isOutputTag;

    private boolean isBounded;

    public AlgorithmFlow() {
        iterationStageList = new ArrayList<>();
        beforeIterationStageList = new ArrayList<>();
        afterIterationStageList = new ArrayList<>();
        isBounded = true;
    }

    public AlgorithmFlow(boolean isBounded) {
        this();
        this.isBounded = isBounded;
    }

    protected MLData processStage(MLData inputData, List<BaseComponent> stages) {
        for (BaseComponent stage : stages) {
            CommonComponent commonStage = (CommonComponent) stage;
            inputData = commonStage.apply(inputData);
        }
        return inputData;
    }

    public AlgorithmFlow startIteration(
            String[] feedbackDataNames, String[] inputDataNames, boolean isReplayData) {
        this.isBeforeIterationStage = false;
        this.isInIterationStage = true;
        this.inputDataNames = inputDataNames;
        this.feedbackDataNames = feedbackDataNames;
        this.isReplayData = isReplayData;
        return this;
    }

    /** Sets the criteria of termination. */
    public AlgorithmFlow endIteration(String[] outputNames, boolean isOutputTag) {
        this.isInIterationStage = false;
        this.outputNames = outputNames;
        if (isOutputTag) {
            this.isOutputTag = isOutputTag;
        }
        return this;
    }

    public AlgorithmFlow startServerIteration(MLSession mlSession, ModelUpdater modelUpdater) {
        this.serverIteration = new ServerIteration<>(mlSession);
        this.serverIteration.modelUpdater = modelUpdater;
        this.isBeforeIterationStage = false;
        this.isInPsIterationStage = true;
        return this;
    }

    @SuppressWarnings("unchecked")
    public AlgorithmFlow endServerIteration(SerializableFunction shouldTerminate) {
        serverIteration.stopIteration(shouldTerminate);
        this.isInPsIterationStage = false;
        this.beforeIterationStageList.add(serverIteration);
        return this;
    }

    /** Adds a stage into the stage list. */
    public AlgorithmFlow add(BaseComponent stage) {
        if (isInIterationStage) {
            iterationStageList.add(stage);
        } else if (isInPsIterationStage) {
            serverIteration.add(stage);
        } else if (isBeforeIterationStage) {
            beforeIterationStageList.add(stage);
        } else {
            afterIterationStageList.add(stage);
        }
        return this;
    }

    @Override
    public MLData apply(MLData inputData) {

        if (this.beforeIterationStageList.size() != 0) {
            inputData = processStage(inputData, beforeIterationStageList);
        }
        if (this.iterationStageList.size() != 0) {
            List<DataStream<?>> modelDataStreams =
                    inputData.slice(feedbackDataNames).getDataStreams();
            List<DataStream<?>> inputDataStreams = inputData.slice(inputDataNames).getDataStreams();

            if (isBounded) {
                inputData =
                        new MLData(
                                Iterations.iterateBoundedStreamsUntilTermination(
                                        DataStreamList.of(
                                                modelDataStreams.toArray(new DataStream[0])),
                                        isReplayData
                                                ? ReplayableDataStreamList.replay(
                                                        inputDataStreams.toArray(new DataStream[0]))
                                                : ReplayableDataStreamList.notReplay(
                                                        inputDataStreams.toArray(
                                                                new DataStream[0])),
                                        IterationConfig.newBuilder()
                                                .build(), // todo : ALL_ROUND set param
                                        new TrainIterationBody(
                                                this,
                                                isBounded,
                                                isOutputTag,
                                                outputNames,
                                                feedbackDataNames,
                                                inputDataNames)));
            } else {
                inputData =
                        new MLData(
                                Iterations.iterateUnboundedStreams(
                                        DataStreamList.of(
                                                modelDataStreams.toArray(new DataStream[0])),
                                        DataStreamList.of(
                                                inputDataStreams.toArray(new DataStream[0])),
                                        new TrainIterationBody(
                                                this,
                                                isBounded,
                                                isOutputTag,
                                                outputNames,
                                                feedbackDataNames,
                                                inputDataNames)));
            }
        }
        if (this.afterIterationStageList.size() != 0) {
            inputData = processStage(inputData, afterIterationStageList);
        }
        return inputData;
    }

    /** The iteration implementation for training process. */
    private static class TrainIterationBody implements IterationBody {
        private final AlgorithmFlow algorithmFlow;
        private final boolean isBound;
        private final boolean isOutputTag;
        private final String[] outputNames;
        String[] feedbackNames;
        String[] inputDataNames;

        public TrainIterationBody(
                AlgorithmFlow algorithmFlow,
                boolean isBound,
                boolean isOutputTag,
                String[] outputNames,
                String[] feedbackNames,
                String[] inputDataNames) {
            this.algorithmFlow = algorithmFlow;
            this.isBound = isBound;
            this.isOutputTag = isOutputTag;
            this.outputNames = outputNames;
            this.feedbackNames = feedbackNames;
            this.inputDataNames = inputDataNames;
        }

        @Override
        @SuppressWarnings("unchecked")
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {

            MLData mlData =
                    new MLData(
                            dataStreams.getDataStreams().toArray(new DataStream<?>[0]),
                            inputDataNames);
            mlData.merge(
                    new MLData(
                            variableStreams.getDataStreams().toArray(new DataStream<?>[0]),
                            feedbackNames));
            OutputTag<?> outputTag = null;
            for (BaseComponent stage : algorithmFlow.iterationStageList) {
                CommonComponent commonStage = (CommonComponent) stage;
                if (stage instanceof CoTransformComponent) {
                    outputTag = ((CoTransformComponent<?, ?, ?>) stage).outputTag;
                }
                mlData = commonStage.apply(mlData);
            }

            if (isBound) {
                DataStream op =
                        isOutputTag
                                ? ((SingleOutputStreamOperator<double[]>)
                                                mlData.get(outputNames[0]))
                                        .getSideOutput(outputTag)
                                : mlData.get(outputNames[0]);
                return new IterationBodyResult(
                        DataStreamList.of(mlData.get(outputNames[1])),
                        DataStreamList.of(op),
                        mlData.get(outputNames[2]));
            } else {
                return new IterationBodyResult(
                        DataStreamList.of(mlData.get(outputNames[0])),
                        DataStreamList.of(mlData.get(outputNames[1])));
            }
        }
    }
}
