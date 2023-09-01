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
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.base.DoubleSerializer;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.ml.classification.fmclassifier.FmClassifierParams;
import org.apache.flink.ml.common.fm.local.LocalAdaDelta;
import org.apache.flink.ml.common.fm.local.LocalAdaGrad;
import org.apache.flink.ml.common.fm.local.LocalAdam;
import org.apache.flink.ml.common.fm.local.LocalFtrl;
import org.apache.flink.ml.common.fm.local.LocalMomentum;
import org.apache.flink.ml.common.fm.local.LocalRMSProp;
import org.apache.flink.ml.common.fm.local.LocalSGD;
import org.apache.flink.ml.common.fm.optim.AdaDelta;
import org.apache.flink.ml.common.fm.optim.AdaGrad;
import org.apache.flink.ml.common.fm.optim.Adam;
import org.apache.flink.ml.common.fm.optim.FTRL;
import org.apache.flink.ml.common.fm.optim.Momentum;
import org.apache.flink.ml.common.fm.optim.RMSProp;
import org.apache.flink.ml.common.fm.optim.SGD;
import org.apache.flink.ml.common.ps.iterations.AllReduceStage;
import org.apache.flink.ml.common.ps.iterations.IterationStageList;
import org.apache.flink.ml.common.ps.iterations.ProcessStage;
import org.apache.flink.ml.common.ps.iterations.PullStage;
import org.apache.flink.ml.common.ps.iterations.PushStage;
import org.apache.flink.ml.common.ps.updater.ModelUpdater;
import org.apache.flink.ml.common.ps.utils.TrainingUtils;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.types.Row;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Fm train. */
public class BaseFmTrain implements FmClassifierParams<BaseFmTrain> {

    private final LossFunction lossFunction;
    private final Map<Param<?>, Object> params;
    private final boolean isReg;
    public static final Logger LOG = LoggerFactory.getLogger(BaseFmTrain.class);
    public static final double EPS = 1.0e-8;

    public BaseFmTrain(boolean isReg, Map<Param<?>, Object> params) {
        if (isReg) {
            double minTarget = -1.0e20;
            double maxTarget = 1.0e20;
            double d = maxTarget - minTarget;
            d = Math.max(d, 1.0);
            maxTarget = maxTarget + d * 0.2;
            minTarget = minTarget - d * 0.2;
            this.lossFunction = new SquareLoss(maxTarget, minTarget);
        } else {
            this.lossFunction = new LogitLoss();
        }
        this.params = params;
        this.isReg = isReg;
    }

    public DataStream<FmModelData> train(DataStream<Row> input) {
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

        DataStream<FmSample> trainData =
                input.rebalance()
                        .map(
                                (MapFunction<Row, FmSample>)
                                        dataPoint -> {
                                            double weight =
                                                    weightCol == null
                                                            ? 1.0
                                                            : ((Number)
                                                                            dataPoint.getFieldAs(
                                                                                    weightCol))
                                                                    .doubleValue();
                                            double label =
                                                    ((Number) dataPoint.getFieldAs(labelCol))
                                                            .doubleValue();
                                            SparseVector vec;
                                            if (dataPoint.getField(featureCol)
                                                    instanceof SparseVector) {
                                                vec = dataPoint.getFieldAs(featureCol);
                                                long[] longIndices = new long[vec.indices.length];
                                                for (int i = 0; i < longIndices.length; ++i) {
                                                    longIndices[i] = vec.indices[i];
                                                }
                                                Tuple2<long[], double[]> features =
                                                        Tuple2.of(longIndices, vec.values);

                                                return new FmSample(features, label, weight);
                                            } else if (dataPoint.getField(featureCol)
                                                    instanceof Tuple2) {
                                                return new FmSample(
                                                        dataPoint.getFieldAs(featureCol),
                                                        label,
                                                        weight);
                                            } else {
                                                throw new RuntimeException(
                                                        "feature type not support yet.");
                                            }
                                        });

        final double initStd = getInitStdEv();
        final double tol = getTol();
        final double learnRate = getLearnRate();
        final int maxIter = getMaxIter();
        final double gamma = getGamma();
        final double beta1 = getBeta1();
        final double beta2 = getBeta2();
        String method = getMethod().toUpperCase();
        FmMLSession mlSession = new FmMLSession(getGlobalBatchSize());

        ModelUpdater<Tuple2<Long, double[]>> fmUpdater;
        ProcessStage<FmMLSession> computeIndices;
        ProcessStage<FmMLSession> computeLocalVariable;
        switch (Method.valueOf(method)) {
            case ADAGRAD:
                fmUpdater = new AdaGrad(dim, learnRate, initStd);
                computeIndices = new ComputeFmIndices(dim[1] + dim[2]);
                computeLocalVariable = new ComputeFmGradients(lossFunction, dim, regular);
                break;
            case FTRL:
                fmUpdater = new FTRL(dim, initStd, getL1(), getL2(), getAlpha(), getBeta());
                computeIndices = new ComputeFmIndices(dim[1] + dim[2]);
                computeLocalVariable = new ComputeFmGradients(lossFunction, dim, regular);
                break;
            case SGD:
                fmUpdater = new SGD(dim, learnRate, initStd);
                computeIndices = new ComputeFmIndices(dim[1] + dim[2]);
                computeLocalVariable = new ComputeFmGradients(lossFunction, dim, regular);
                break;
            case ADAM:
                fmUpdater = new Adam(dim, learnRate, initStd, beta1, beta2);
                computeIndices = new ComputeFmIndices(dim[1] + dim[2]);
                computeLocalVariable = new ComputeFmGradients(lossFunction, dim, regular);
                break;
            case MOMENTUM:
                fmUpdater = new Momentum(dim, learnRate, initStd, gamma);
                computeIndices = new ComputeFmIndices(dim[1] + dim[2]);
                computeLocalVariable = new ComputeFmGradients(lossFunction, dim, regular);
                break;
            case RMSPROP:
                fmUpdater = new RMSProp(dim, learnRate, initStd, gamma);
                computeIndices = new ComputeFmIndices(dim[1] + dim[2]);
                computeLocalVariable = new ComputeFmGradients(lossFunction, dim, regular);
                break;
            case ADADELTA:
                fmUpdater = new AdaDelta(dim, initStd, gamma);
                computeIndices = new ComputeFmIndices(dim[1] + dim[2]);
                computeLocalVariable = new ComputeFmGradients(lossFunction, dim, regular);
                break;
            case ADAGRAD_AVG:
                int modelSize = 2 * (dim[1] + dim[2]) + 1;
                fmUpdater = new UpdateFmFactors(dim[1] + dim[2], modelSize, initStd);
                computeIndices = new ComputeFmIndices(modelSize);
                computeLocalVariable = new LocalAdaGrad(lossFunction, dim, regular, learnRate);
                break;
            case RMSPROP_AVG:
                modelSize = 2 * (dim[1] + dim[2]) + 1;
                fmUpdater = new UpdateFmFactors(dim[1] + dim[2], modelSize, initStd);
                computeIndices = new ComputeFmIndices(modelSize);
                computeLocalVariable =
                        new LocalRMSProp(lossFunction, dim, regular, learnRate, gamma);
                break;
            case MOMENTUM_AVG:
                modelSize = 2 * (dim[1] + dim[2]) + 1;
                fmUpdater = new UpdateFmFactors(dim[1] + dim[2], modelSize, initStd);
                computeIndices = new ComputeFmIndices(modelSize);
                computeLocalVariable =
                        new LocalMomentum(lossFunction, dim, regular, learnRate, gamma);
                break;
            case FTRL_AVG:
                modelSize = 3 * (dim[1] + dim[2]) + 1;
                fmUpdater = new UpdateFmFactors(dim[1] + dim[2], modelSize, initStd);
                computeIndices = new ComputeFmIndices(modelSize);
                computeLocalVariable =
                        new LocalFtrl(
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
                        new LocalAdam(lossFunction, dim, regular, learnRate, beta1, beta2);
                break;
            case ADADELTA_AVG:
                modelSize = 3 * (dim[1] + dim[2]) + 1;
                fmUpdater = new UpdateFmFactors(dim[1] + dim[2], modelSize, initStd);
                computeIndices = new ComputeFmIndices(modelSize);
                computeLocalVariable = new LocalAdaDelta(lossFunction, dim, regular, gamma);
                break;
            case SGD_AVG:
                modelSize = dim[1] + dim[2] + 1;
                fmUpdater = new UpdateFmFactors(dim[1] + dim[2], modelSize, initStd);
                computeIndices = new ComputeFmIndices(modelSize);
                computeLocalVariable = new LocalSGD(lossFunction, dim, regular, learnRate);
                break;
            default:
                throw new RuntimeException("not support yet.");
        }

        IterationStageList<FmMLSession> iterationStages =
                new IterationStageList<>(mlSession)
                        .addStage(computeIndices)
                        .addStage(new PullStage(() -> mlSession.indices, () -> mlSession.values))
                        .addStage(computeLocalVariable)
                        .addStage(new PushStage(() -> mlSession.indices, () -> mlSession.values))
                        .addStage(
                                new AllReduceStage<>(
                                        () -> mlSession.localLoss,
                                        () -> mlSession.globalLoss,
                                        (ReduceFunction<Double[]>) BaseFmTrain::sumDoubleArray,
                                        DoubleSerializer.INSTANCE,
                                        1))
                        .setTerminationCriteria(
                                o -> {
                                    int numMiniBatch = o.globalLoss[2].intValue() / o.numWorkers;
                                    if ((o.iterationId - 1) % numMiniBatch == 0
                                            || o.iterationId == maxIter * numMiniBatch) {
                                        System.out.println(
                                                "Loss at epoch-"
                                                        + (o.iterationId / numMiniBatch)
                                                        + " is: "
                                                        + o.globalLoss[0] / o.globalLoss[1]
                                                        + ".\n");
                                    }
                                    return o.iterationId >= maxIter * numMiniBatch
                                            || o.globalLoss[0] < tol;
                                });

        int parallelism = trainData.getParallelism();
        DataStreamList resultList =
                TrainingUtils.train(
                        trainData,
                        iterationStages,
                        new TupleTypeInfo<>(
                                Types.LONG,
                                PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO),
                        fmUpdater,
                        Math.max(1, parallelism / 2));

        DataStream<Tuple2<Long, double[]>> optResult = resultList.get(0);

        return optResult
                .transform(
                        "generateModelData",
                        TypeInformation.of(FmModelData.class),
                        new GenerateModelData(dim, isReg))
                .name("generateModelData");
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return params;
    }

    /** Generates the ModelData from the results of iteration. */
    private static class GenerateModelData extends AbstractStreamOperator<FmModelData>
            implements OneInputStreamOperator<Tuple2<Long, double[]>, FmModelData>,
                    BoundedOneInput {

        private final List<Tuple2<Long, float[]>> factors = new ArrayList<>();
        private final int[] dim;
        private final boolean isReg;

        public GenerateModelData(int[] dim, boolean isReg) {
            this.dim = dim;
            this.isReg = isReg;
        }

        @Override
        public void endInput() throws Exception {
            LOG.info("Generates model   ... " + System.currentTimeMillis());
            output.collect(new StreamRecord<>(new FmModelData(factors, dim, isReg)));
        }

        @Override
        public void processElement(StreamRecord<Tuple2<Long, double[]>> streamRecord)
                throws Exception {
            Tuple2<Long, double[]> t2 = streamRecord.getValue();
            float[] factor = new float[t2.f1.length];
            for (int i = 0; i < factor.length; ++i) {
                factor[i] = (float) t2.f1[i];
            }
            factors.add(Tuple2.of(t2.f0, factor));
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
        private final double maxTarget;
        private final double minTarget;

        public SquareLoss(double maxTarget, double minTarget) {
            this.maxTarget = maxTarget;
            this.minTarget = minTarget;
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

    private static Double[] sumDoubleArray(Double[] array1, Double[] array2) {
        for (int i = 0; i < array1.length; i++) {
            array2[i] += array1[i];
        }
        return array2;
    }

    /** loss function for binary classification task. */
    public static final class LogitLoss implements LossFunction {

        @Override
        public double loss(double yTruth, double y) { // yTruth in {0, 1}
            double logit;
            if (y < -37) {
                logit = EPS;
            } else if (y > 34) {
                logit = 1.0 - EPS;
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
