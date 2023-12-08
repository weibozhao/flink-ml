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

package org.apache.flink.ml.regression.fmregressor;

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.typeutils.base.DoubleSerializer;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.fm.BaseFmTrain;
import org.apache.flink.ml.common.fm.BaseFmTrain.LossFunction;
import org.apache.flink.ml.common.fm.BaseFmTrain.SquareLoss;
import org.apache.flink.ml.common.fm.BaseFmTrain.Termination;
import org.apache.flink.ml.common.fm.BaseFmTrain.TransformSample;
import org.apache.flink.ml.common.fm.ComputeFmGradients;
import org.apache.flink.ml.common.fm.ComputeFmIndices;
import org.apache.flink.ml.common.fm.FmMLSession;
import org.apache.flink.ml.common.fm.optim.minibatch.AdaGrad;
import org.apache.flink.ml.common.ps.api.AlgorithmFlow;
import org.apache.flink.ml.common.ps.api.MLData;
import org.apache.flink.ml.common.ps.api.MLData.MLDataFunction;
import org.apache.flink.ml.common.ps.iterations.PsAllReduceComponent;
import org.apache.flink.ml.common.ps.iterations.PullComponent;
import org.apache.flink.ml.common.ps.iterations.PushComponent;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/** An Estimator which implements the fm regressor algorithm. */
public class FmRegressor
        implements Estimator<FmRegressor, FmRegressorModel>, FmRegressorParams<FmRegressor> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public FmRegressor() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public FmRegressorModel fit(Table... inputs) {
        MLData mlData = MLData.of(inputs, new String[] {"data"});

        Preconditions.checkArgument(inputs.length == 1);
        int[] dim = new int[3];
        for (int i = 0; i < 3; i++) {
            dim[i] = Integer.parseInt(getDim().split(",")[i].trim());
        }
        double[] regular = new double[3];
        for (int i = 0; i < 3; i++) {
            regular[i] = Double.parseDouble(getLambda().split(",")[i].trim());
        }

        LossFunction lossFunction = new SquareLoss();

        FmMLSession mlSession = new FmMLSession(getGlobalBatchSize());

        AlgorithmFlow algorithmFlow =
                new AlgorithmFlow(true)
                        .add(new MLDataFunction("rebalance"))
                        .add(
                                new MLDataFunction(
                                        "map",
                                        new TransformSample(
                                                getWeightCol(), getLabelCol(), getFeaturesCol())))
                        .startServerIteration(
                                mlSession, new AdaGrad(dim, getLearnRate(), getInitStdEv()))
                        .add(new ComputeFmIndices(dim[1] + dim[2]))
                        .add(new PullComponent(() -> mlSession.indices, () -> mlSession.values))
                        .add(new ComputeFmGradients(lossFunction, dim, regular))
                        .add(new PushComponent(() -> mlSession.indices, () -> mlSession.values))
                        .add(
                                new PsAllReduceComponent<>(
                                        () -> mlSession.localLoss,
                                        () -> mlSession.globalLoss,
                                        (ReduceFunction<Double[]>) BaseFmTrain::sumDoubleArray,
                                        DoubleSerializer.INSTANCE,
                                        1))
                        .endServerIteration(new Termination(getMaxIter(), getTol()))
                        .add(new MLDataFunction("mapPartition", new GenerateModelData(dim, true)));

        MLData modelData = algorithmFlow.apply(mlData);
        FmRegressorModel model = new FmRegressorModel().setModelData(modelData.getTables());
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static FmRegressor load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
