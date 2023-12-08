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

package org.apache.flink.ml.common.optimizer;

import org.apache.flink.annotation.Internal;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.common.lossfunc.LossFunc;
import org.apache.flink.ml.common.ps.api.AlgorithmFlow;
import org.apache.flink.ml.common.ps.api.CoTransformComponent;
import org.apache.flink.ml.common.ps.api.MLData;
import org.apache.flink.ml.common.ps.api.MLData.MLDataFunction;
import org.apache.flink.ml.common.ps.api.TerminateComponent;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.regression.linearregression.LinearRegression;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * Stochastic Gradient Descent (SGD) is the mostly wide-used optimizer for optimizing machine
 * learning models. It iteratively makes small adjustments to the machine learning model according
 * to the gradient at each step, to decrease the error of the model.
 *
 * <p>See https://en.wikipedia.org/wiki/Stochastic_gradient_descent.
 */
@Internal
public class SGD implements Optimizer {
    /** Params for SGD optimizer. */
    private final SGDParams params;

    public SGD(
            int maxIter,
            double learningRate,
            int globalBatchSize,
            double tol,
            double reg,
            double elasticNet) {
        this.params = new SGDParams(maxIter, learningRate, globalBatchSize, tol, reg, elasticNet);
    }

    @Override
    public DataStream<DenseVector> optimize(MLData mlData, LossFunc lossFunc) {
        final OutputTag<DenseVector> modelDataOutputTag =
                new OutputTag<DenseVector>("MODEL_OUTPUT") {};

        AlgorithmFlow algorithmFlow =
                new AlgorithmFlow(true)
                        .add(new MLDataFunction("broadcast").input("modelName").output("modelName"))
                        .add(
                                new MLDataFunction(
                                                "map",
                                                (MapFunction<DenseVector, double[]>)
                                                        denseVector -> denseVector.values)
                                        .returns(TypeInformation.of(double[].class)))
                        .startIteration(new String[] {"modelName"}, new String[] {"data"}, false)
                        .add(
                                new CacheDataAndDoTrain(lossFunc, params)
                                        .with("modelName")
                                        .input("data")
                                        .output("grad")
                                        .withOutputTag(modelDataOutputTag)
                                        .withOutType(
                                                PrimitiveArrayTypeInfo
                                                        .DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO))
                        .add(
                                new MLDataFunction("allReduce")
                                        .input("grad")
                                        .output("gradAll")
                                        .isForEachRound(true))
                        .add(
                                new MLDataFunction(
                                                "map",
                                                (MapFunction<double[], Double>)
                                                        value ->
                                                                value[value.length - 1]
                                                                        / value[value.length - 2])
                                        .returns(TypeInformation.of(Double.class))
                                        .input("gradAll")
                                        .output("Integer"))
                        .add(
                                new TerminateOnMaxComponent(params.maxIter, params.tol)
                                        .input("Integer")
                                        .output("last"))
                        .endIteration(new String[] {"grad", "gradAll", "last"}, true);

        mlData = algorithmFlow.apply(mlData);

        return mlData.get(0);
    }

    /** Comments. */
    public static class TerminateOnMaxComponent extends TerminateComponent<Double, Integer> {

        private final int maxIter;

        private final double tol;

        private double loss = Double.MAX_VALUE;

        public TerminateOnMaxComponent(Integer maxIter, Double tol) {
            this.maxIter = maxIter;
            this.tol = tol;
        }

        public TerminateOnMaxComponent(Double tol) {
            this.maxIter = Integer.MAX_VALUE;
            this.tol = tol;
        }

        @Override
        public void flatMap(Double value, Collector<Integer> out) {
            Preconditions.checkArgument(
                    Double.compare(loss, Double.MAX_VALUE) == 0,
                    "Each epoch should contain only one loss value.");
            loss = value;
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<Integer> collector) {
            if ((epochWatermark + 1) < maxIter && loss > tol) {
                collector.collect(0);
            }
            loss = Double.MAX_VALUE;
        }

        @Override
        public void onIterationTerminated(Context context, Collector<Integer> collector) {}
    }

    /**
     * A stream operator that caches the training data in the first iteration and updates the model
     * iteratively. The first input is the training data, and the second input is the initial model
     * data or feedback of model update, totalWeight, and totalLoss.
     */
    private static class CacheDataAndDoTrain
            extends CoTransformComponent<LabeledPointWithWeight, double[], double[]> {
        /** Optimizer-related parameters. */
        private final SGDParams params;

        /** The loss function to optimize. */
        private final LossFunc lossFunc;

        /** The outputTag to output the model data when iteration ends. */

        /** The cached training data. */
        private List<LabeledPointWithWeight> trainData;

        private ListState<LabeledPointWithWeight> trainDataState;

        /** The start index (offset) of the next mini-batch data for training. */
        private int nextBatchOffset = 0;

        private ListState<Integer> nextBatchOffsetState;

        /** The model coefficient. */
        private DenseVector coefficient;

        private ListState<DenseVector> coefficientState;

        /** The dimension of the coefficient. */
        private int coefficientDim;

        /**
         * The double array to sync among all workers. For example, when training {@link
         * LinearRegression}, this double array consists of [modelUpdate, totalWeight, totalLoss].
         */
        private double[] feedbackArray;

        private ListState<double[]> feedbackArrayState;

        /** The batch size on this partition. */
        private int localBatchSize;

        private CacheDataAndDoTrain(LossFunc lossFunc, SGDParams params) {
            this.lossFunc = lossFunc;
            this.params = params;
        }

        @Override
        public void open() {
            int numTasks = getRuntimeContext().getNumberOfParallelSubtasks();
            int taskId = getRuntimeContext().getIndexOfThisSubtask();
            localBatchSize = params.globalBatchSize / numTasks;
            if (params.globalBatchSize % numTasks > taskId) {
                localBatchSize++;
            }
        }

        private double getTotalWeight() {
            return feedbackArray[coefficientDim];
        }

        private void setTotalWeight(double totalWeight) {
            feedbackArray[coefficientDim] = totalWeight;
        }

        private double getTotalLoss() {
            return feedbackArray[coefficientDim + 1];
        }

        private void setTotalLoss(double totalLoss) {
            feedbackArray[coefficientDim + 1] = totalLoss;
        }

        private void updateModel() {
            if (getTotalWeight() > 0) {
                BLAS.axpy(
                        -params.learningRate / getTotalWeight(),
                        new DenseVector(feedbackArray),
                        coefficient,
                        coefficientDim);
                double regLoss =
                        RegularizationUtils.regularize(
                                coefficient, params.reg, params.elasticNet, params.learningRate);
                setTotalLoss(getTotalLoss() + regLoss);
            }
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<double[]> collector)
                throws Exception {
            if (epochWatermark == 0) {
                coefficient = new DenseVector(feedbackArray);
                coefficientDim = coefficient.size();
                feedbackArray = new double[coefficient.size() + 2];
            } else {
                updateModel();
            }

            if (trainData == null) {
                trainData = IteratorUtils.toList(trainDataState.get().iterator());
            }

            // TODO: supports efficient shuffle of training set on each partition.
            if (trainData.size() > 0) {
                List<LabeledPointWithWeight> miniBatchData =
                        trainData.subList(
                                nextBatchOffset,
                                Math.min(nextBatchOffset + localBatchSize, trainData.size()));
                nextBatchOffset += localBatchSize;
                nextBatchOffset = nextBatchOffset >= trainData.size() ? 0 : nextBatchOffset;

                // Does the training.
                Arrays.fill(feedbackArray, 0);
                double totalLoss = 0;
                double totalWeight = 0;
                DenseVector cumGradientsWrapper = new DenseVector(feedbackArray);
                for (LabeledPointWithWeight dataPoint : miniBatchData) {
                    totalLoss += lossFunc.computeLoss(dataPoint, coefficient);
                    lossFunc.computeGradient(dataPoint, coefficient, cumGradientsWrapper);
                    totalWeight += dataPoint.getWeight();
                }
                setTotalLoss(totalLoss);
                setTotalWeight(totalWeight);

                collector.collect(feedbackArray);
            }
        }

        @Override
        public void onIterationTerminated(Context context, Collector<double[]> collector) {
            trainDataState.clear();
            if (getRuntimeContext().getIndexOfThisSubtask() == 0) {
                updateModel();
                context.output(this.outputTag, coefficient);
            }
        }

        @Override
        public void processElement1(StreamRecord<LabeledPointWithWeight> streamRecord)
                throws Exception {
            trainDataState.add(streamRecord.getValue());
        }

        @Override
        public void processElement2(StreamRecord<double[]> streamRecord) {
            feedbackArray = streamRecord.getValue();
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            coefficientState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "coefficientState", DenseVectorTypeInfo.INSTANCE));
            OperatorStateUtils.getUniqueElement(coefficientState, "coefficientState")
                    .ifPresent(x -> coefficient = x);
            if (coefficient != null) {
                coefficientDim = coefficient.size();
            }

            feedbackArrayState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "feedbackArrayState",
                                            PrimitiveArrayTypeInfo
                                                    .DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO));
            OperatorStateUtils.getUniqueElement(feedbackArrayState, "feedbackArrayState")
                    .ifPresent(x -> feedbackArray = x);

            trainDataState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "trainDataState",
                                            TypeInformation.of(LabeledPointWithWeight.class)));

            nextBatchOffsetState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "nextBatchOffsetState", BasicTypeInfo.INT_TYPE_INFO));
            nextBatchOffset =
                    OperatorStateUtils.getUniqueElement(
                                    nextBatchOffsetState, "nextBatchOffsetState")
                            .orElse(0);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            coefficientState.clear();
            if (coefficient != null) {
                coefficientState.add(coefficient);
            }

            feedbackArrayState.clear();
            if (feedbackArray != null) {
                feedbackArrayState.add(feedbackArray);
            }

            nextBatchOffsetState.clear();
            nextBatchOffsetState.add(nextBatchOffset);
        }
    }

    /** Parameters for {@link SGD}. */
    private static class SGDParams implements Serializable {
        public final int maxIter;
        public final double learningRate;
        public final int globalBatchSize;
        public final double tol;
        public final double reg;
        public final double elasticNet;

        private SGDParams(
                int maxIter,
                double learningRate,
                int globalBatchSize,
                double tol,
                double reg,
                double elasticNet) {
            this.maxIter = maxIter;
            this.learningRate = learningRate;
            this.globalBatchSize = globalBatchSize;
            this.tol = tol;
            this.reg = reg;
            this.elasticNet = elasticNet;
        }
    }
}
