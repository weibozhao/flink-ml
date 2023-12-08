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

package org.apache.flink.ml.common.fm;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.typeutils.base.DoubleSerializer;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.fm.optim.avg.AdaDelta;
import org.apache.flink.ml.common.fm.optim.avg.AdaGrad;
import org.apache.flink.ml.common.fm.optim.avg.Adam;
import org.apache.flink.ml.common.fm.optim.avg.Ftrl;
import org.apache.flink.ml.common.fm.optim.avg.Momentum;
import org.apache.flink.ml.common.fm.optim.avg.RMSProp;
import org.apache.flink.ml.common.fm.optim.avg.SGD;
import org.apache.flink.ml.common.fm.optim.minibatch.FTRL;
import org.apache.flink.ml.common.param.HasFeaturesCol;
import org.apache.flink.ml.common.ps.api.AlgorithmFlow;
import org.apache.flink.ml.common.ps.api.MLData;
import org.apache.flink.ml.common.ps.api.MLData.MLDataFunction;
import org.apache.flink.ml.common.ps.iterations.ProcessComponent;
import org.apache.flink.ml.common.ps.iterations.PsAllReduceComponent;
import org.apache.flink.ml.common.ps.iterations.PullComponent;
import org.apache.flink.ml.common.ps.iterations.PushComponent;
import org.apache.flink.ml.common.ps.updater.ModelUpdater;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.function.SerializableFunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Fm train. */
public class BaseFmTrain implements FmCommonParams<BaseFmTrain>, HasFeaturesCol<BaseFmTrain> {

    private final LossFunction lossFunction;
    private final Map<Param<?>, Object> params;
    private final boolean isReg;
    public static final Logger LOG = LoggerFactory.getLogger(BaseFmTrain.class);
    public static final double EPS = 1.0e-8;

    public BaseFmTrain(boolean isReg, Map<Param<?>, Object> params) {
        if (isReg) {
            this.lossFunction = new SquareLoss();
        } else {
            this.lossFunction = new LogitLoss();
        }
        this.params = params;
        this.isReg = isReg;
    }

    public MLData train(MLData input) {
        String weightCol = getWeightCol();
        String labelCol = getLabelCol();
        String featureCol = getFeaturesCol();
        String dimStr = getDim();
        String regularStr = getLambda();

        int[] dim = new int[3];
        for (int i = 0; i < 3; i++) {
            dim[i] = Integer.parseInt(dimStr.split(",")[i].trim());
        }

        double[] regular = new double[3];
        for (int i = 0; i < 3; i++) {
            regular[i] = Double.parseDouble(regularStr.split(",")[i].trim());
        }

        final double initStd = getInitStdEv();
        final double tol = getTol();
        final double learnRate = getLearnRate();
        final int maxIter = getMaxIter();
        final double gamma = getGamma();
        final double beta1 = getBeta1();
        final double beta2 = getBeta2();
        String method = getMethod().toUpperCase();
        FmMLSession mlSession = new FmMLSession(getGlobalBatchSize());

        ModelUpdater fmUpdater;
        ProcessComponent<FmMLSession> computeIndices;
        ProcessComponent<FmMLSession> computeLocalVariable;
        switch (Method.valueOf(method)) {
            case ADAGRAD:
                fmUpdater =
                        new org.apache.flink.ml.common.fm.optim.minibatch.AdaGrad(
                                dim, learnRate, initStd);
                computeIndices = new ComputeFmIndices(dim[1] + dim[2]);
                computeLocalVariable = new ComputeFmGradients(lossFunction, dim, regular);
                break;
            case FTRL:
                fmUpdater = new FTRL(dim, initStd, getL1(), getL2(), getAlpha(), getBeta());
                computeIndices = new ComputeFmIndices(dim[1] + dim[2]);
                computeLocalVariable = new ComputeFmGradients(lossFunction, dim, regular);
                break;
            case SGD:
                fmUpdater =
                        new org.apache.flink.ml.common.fm.optim.minibatch.SGD(
                                dim, learnRate, initStd);
                computeIndices = new ComputeFmIndices(dim[1] + dim[2]);
                computeLocalVariable = new ComputeFmGradients(lossFunction, dim, regular);
                break;
            case ADAM:
                fmUpdater =
                        new org.apache.flink.ml.common.fm.optim.minibatch.Adam(
                                dim, learnRate, initStd, beta1, beta2);
                computeIndices = new ComputeFmIndices(dim[1] + dim[2]);
                computeLocalVariable = new ComputeFmGradients(lossFunction, dim, regular);
                break;
            case MOMENTUM:
                fmUpdater =
                        new org.apache.flink.ml.common.fm.optim.minibatch.Momentum(
                                dim, learnRate, initStd, gamma);
                computeIndices = new ComputeFmIndices(dim[1] + dim[2]);
                computeLocalVariable = new ComputeFmGradients(lossFunction, dim, regular);
                break;
            case RMSPROP:
                fmUpdater =
                        new org.apache.flink.ml.common.fm.optim.minibatch.RMSProp(
                                dim, learnRate, initStd, gamma);
                computeIndices = new ComputeFmIndices(dim[1] + dim[2]);
                computeLocalVariable = new ComputeFmGradients(lossFunction, dim, regular);
                break;
            case ADADELTA:
                fmUpdater =
                        new org.apache.flink.ml.common.fm.optim.minibatch.AdaDelta(
                                dim, initStd, gamma);
                computeIndices = new ComputeFmIndices(dim[1] + dim[2]);
                computeLocalVariable = new ComputeFmGradients(lossFunction, dim, regular);
                break;
            case ADAGRAD_AVG:
                int modelSize = 2 * (dim[1] + dim[2]) + 1;
                fmUpdater = new UpdateFmFactors(dim[1] + dim[2], modelSize, initStd);
                computeIndices = new ComputeFmIndices(modelSize);
                computeLocalVariable = new AdaGrad(lossFunction, dim, regular, learnRate);
                break;
            case RMSPROP_AVG:
                modelSize = 2 * (dim[1] + dim[2]) + 1;
                fmUpdater = new UpdateFmFactors(dim[1] + dim[2], modelSize, initStd);
                computeIndices = new ComputeFmIndices(modelSize);
                computeLocalVariable = new RMSProp(lossFunction, dim, regular, learnRate, gamma);
                break;
            case MOMENTUM_AVG:
                modelSize = 2 * (dim[1] + dim[2]) + 1;
                fmUpdater = new UpdateFmFactors(dim[1] + dim[2], modelSize, initStd);
                computeIndices = new ComputeFmIndices(modelSize);
                computeLocalVariable = new Momentum(lossFunction, dim, regular, learnRate, gamma);
                break;
            case FTRL_AVG:
                modelSize = 3 * (dim[1] + dim[2]) + 1;
                fmUpdater = new UpdateFmFactors(dim[1] + dim[2], modelSize, initStd);
                computeIndices = new ComputeFmIndices(modelSize);
                computeLocalVariable =
                        new Ftrl(
                                lossFunction,
                                dim,
                                regular,
                                getL1(),
                                getL2(),
                                getAlpha(),
                                getBeta());
                break;
            case ADAM_AVG:
                modelSize = 3 * (dim[1] + dim[2]) + 3;
                fmUpdater = new UpdateFmFactors(dim[1] + dim[2], modelSize, initStd);
                computeIndices = new ComputeFmIndices(modelSize);
                computeLocalVariable =
                        new Adam(lossFunction, dim, regular, learnRate, beta1, beta2);
                break;
            case ADADELTA_AVG:
                modelSize = 3 * (dim[1] + dim[2]) + 1;
                fmUpdater = new UpdateFmFactors(dim[1] + dim[2], modelSize, initStd);
                computeIndices = new ComputeFmIndices(modelSize);
                computeLocalVariable = new AdaDelta(lossFunction, dim, regular, gamma);
                break;
            case SGD_AVG:
                modelSize = dim[1] + dim[2] + 1;
                fmUpdater = new UpdateFmFactors(dim[1] + dim[2], modelSize, initStd);
                computeIndices = new ComputeFmIndices(modelSize);
                computeLocalVariable = new SGD(lossFunction, dim, regular, learnRate);
                break;
            default:
                throw new RuntimeException("not support yet.");
        }

        AlgorithmFlow algorithmFlow =
                new AlgorithmFlow(true)
                        .add(new MLDataFunction("rebalance"))
                        .add(
                                new MLDataFunction(
                                        "map",
                                        new TransformSample(weightCol, labelCol, featureCol)))
                        .startServerIteration(mlSession, fmUpdater)
                        .add(computeIndices)
                        .add(new PullComponent(() -> mlSession.indices, () -> mlSession.values))
                        .add(computeLocalVariable)
                        .add(new PushComponent(() -> mlSession.indices, () -> mlSession.values))
                        .add(
                                new PsAllReduceComponent<>(
                                        () -> mlSession.localLoss,
                                        () -> mlSession.globalLoss,
                                        (ReduceFunction<Double[]>) BaseFmTrain::sumDoubleArray,
                                        DoubleSerializer.INSTANCE,
                                        1))
                        .endServerIteration(new Termination(maxIter, tol))
                        .add(new MLDataFunction("mapPartition", new GenerateModelData(dim, isReg)));

        return algorithmFlow.apply(input);
    }

    /** Comments. */
    public static class TransformSample implements MapFunction<Row, FmSample> {

        private final String weightCol;
        private final String labelCol;
        private final String featureCol;

        public TransformSample(String weightCol, String labelCol, String featureCol) {
            this.weightCol = weightCol;
            this.labelCol = labelCol;
            this.featureCol = featureCol;
        }

        @Override
        public FmSample map(Row row) {
            double weight =
                    weightCol == null ? 1.0 : ((Number) row.getFieldAs(weightCol)).doubleValue();
            double label = ((Number) row.getFieldAs(labelCol)).doubleValue();
            SparseVector vec;
            if (row.getField(featureCol) instanceof SparseVector) {
                vec = row.getFieldAs(featureCol);
                long[] longIndices = new long[vec.indices.length];
                for (int i = 0; i < longIndices.length; ++i) {
                    longIndices[i] = vec.indices[i];
                }
                Tuple2<long[], double[]> features = Tuple2.of(longIndices, vec.values);

                return new FmSample(features, label, weight);
            } else if (row.getField(featureCol) instanceof Tuple2) {
                return new FmSample(row.getFieldAs(featureCol), label, weight);
            } else {
                throw new RuntimeException("feature type not support yet.");
            }
        }
    }

    /** Comments. */
    public static class Termination implements SerializableFunction<FmMLSession, Boolean> {

        private final int maxIter;
        private final double tol;

        public Termination(int maxIter, double tol) {
            this.maxIter = maxIter;
            this.tol = tol;
        }

        @Override
        public Boolean apply(FmMLSession mlSession) {
            int numMiniBatch = mlSession.globalLoss[2].intValue() / mlSession.numWorkers;
            if ((mlSession.iterationId - 1) % numMiniBatch == 0
                    || mlSession.iterationId == maxIter * numMiniBatch) {
                System.out.println(
                        "Loss at epoch-"
                                + (mlSession.iterationId / numMiniBatch)
                                + " is: "
                                + mlSession.globalLoss[0] / mlSession.globalLoss[1]
                                + ".\n");
            }
            return mlSession.iterationId >= maxIter * numMiniBatch || mlSession.globalLoss[0] < tol;
        }
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return params;
    }

    /** Generates the ModelData from the results of iteration. */
    private static class GenerateModelData implements MapPartitionFunction<Object, FmModelData> {

        private final List<Tuple2<Long, float[]>> factors = new ArrayList<>();
        private final int[] dim;
        private final boolean isReg;

        public GenerateModelData(int[] dim, boolean isReg) {
            this.dim = dim;
            this.isReg = isReg;
        }

        @Override
        @SuppressWarnings("unchecked")
        public void mapPartition(Iterable<Object> iterable, Collector<FmModelData> collector) {
            for (Object ele : iterable) {
                Tuple2<Long, double[]> t2 = (Tuple2<Long, double[]>) ele;
                float[] factor = new float[t2.f1.length];
                for (int i = 0; i < factor.length; ++i) {
                    factor[i] = (float) t2.f1[i];
                }
                factors.add(Tuple2.of(t2.f0, factor));
            }
            collector.collect(new FmModelData(factors, dim, isReg));
        }
    }

    /** loss function interface. */
    public interface LossFunction extends Serializable {

        /* Loss function. */
        double loss(double yTruth, double y);

        /* Gradient function. */
        double gradient(double yTruth, double y);
    }

    /** loss function for regression task. */
    public static final class SquareLoss implements LossFunction {
        private double maxTarget;
        private double minTarget;

        public SquareLoss() {
            minTarget = -1.0e20;
            maxTarget = 1.0e20;
            double d = maxTarget - minTarget;
            d = Math.max(d, 1.0);
            maxTarget = maxTarget + d * 0.2;
            minTarget = minTarget - d * 0.2;
        }

        @Override
        public double loss(double yTruth, double y) {
            return (yTruth - y) * (yTruth - y);
        }

        @Override
        public double gradient(double yTruth, double y) {

            // a trick borrowed from libFM
            y = Math.min(y, maxTarget);
            y = Math.max(y, minTarget);

            return 2.0 * (y - yTruth);
        }
    }

    public static Double[] sumDoubleArray(Double[] array1, Double[] array2) {
        for (int i = 0; i < array1.length; i++) {
            array2[i] += array1[i];
        }
        return array2;
    }

    /** loss function for binary classification task. */
    public static final class LogitLoss implements LossFunction {
        private static final double eps = 1.0e-15;

        @Override
        public double loss(double yTruth, double y) { // yTruth in {0, 1}
            double logit;
            if (y < -37) {
                logit = eps;
            } else if (y > 34) {
                logit = 1.0 - eps;
            } else {
                logit = 1.0 / (1.0 + Math.exp(-y));
            }
            if (yTruth < 0.5) {
                return -Math.log(1.0 - logit);
            } else {
                return -Math.log(logit);
            }
        }

        @Override
        public double gradient(double yTruth, double y) {
            return sigmoid(y) - yTruth;
        }

        private double sigmoid(double y) {
            if (y < -37) {
                return 0.0;
            } else if (y > 34) {
                return 1.0;
            } else {
                return 1.0 / (1.0 + Math.exp(-y));
            }
        }
    }

    enum Method {
        ADAGRAD,
        ADAGRAD_AVG,
        FTRL,
        FTRL_AVG,
        SGD,
        SGD_AVG,
        ADAM,
        ADAM_AVG,
        MOMENTUM,
        MOMENTUM_AVG,
        RMSPROP,
        RMSPROP_AVG,
        ADADELTA,
        ADADELTA_AVG,
    }
}
