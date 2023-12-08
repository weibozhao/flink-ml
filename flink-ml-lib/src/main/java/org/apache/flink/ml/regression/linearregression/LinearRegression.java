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

package org.apache.flink.ml.regression.linearregression;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.classification.linearsvc.LinearSVC.ParseSample;
import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.common.lossfunc.LeastSquareLoss;
import org.apache.flink.ml.common.ps.api.AlgorithmFlow;
import org.apache.flink.ml.common.ps.api.MLData;
import org.apache.flink.ml.common.ps.api.MLData.MLDataFunction;
import org.apache.flink.ml.common.ps.api.OptimizerComponent;
import org.apache.flink.ml.common.ps.api.OptimizerComponent.Method;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * An Estimator which implements the linear regression algorithm.
 *
 * <p>See https://en.wikipedia.org/wiki/Linear_regression.
 */
public class LinearRegression
        implements Estimator<LinearRegression, LinearRegressionModel>,
                LinearRegressionParams<LinearRegression> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public LinearRegression() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    @SuppressWarnings({"rawTypes", "ConstantConditions"})
    public LinearRegressionModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        String dataName = "data";
        String modelName = "modelName";
        MLData mlData = MLData.of(inputs, new String[] {dataName});
        AlgorithmFlow algoFlow = new AlgorithmFlow();

        algoFlow.add(
                        new MLDataFunction(
                                "map",
                                new ParseSample(getLabelCol(), getWeightCol(), getFeaturesCol())))
                .add(
                        new MLDataFunction(
                                        "map",
                                        (MapFunction<LabeledPointWithWeight, Integer>)
                                                (x -> x.getFeatures().size()))
                                .output(modelName))
                .add(
                        new MLDataFunction(
                                "reduce",
                                (ReduceFunction<Integer>)
                                        (t0, t1) -> {
                                            Preconditions.checkState(
                                                    t0.equals(t1),
                                                    "The training data should all have same dimensions.");
                                            return t0;
                                        }))
                .add(
                        new MLDataFunction(
                                "map", (MapFunction<Integer, DenseVector>) (DenseVector::new)))
                .add(
                        new OptimizerComponent(paramMap, Method.SGD, LeastSquareLoss.INSTANCE)
                                .withInitModel(modelName)
                                .input(dataName)
                                .output(modelName))
                .add(
                        new MLDataFunction(
                                "map",
                                (MapFunction<DenseVector, LinearRegressionModelData>)
                                        (LinearRegressionModelData::new)));

        LinearRegressionModel model =
                new LinearRegressionModel()
                        .setModelData(algoFlow.apply(mlData).getTable(modelName));
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static LinearRegression load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
