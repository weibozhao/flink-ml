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

package org.apache.flink.ml.classification.logisticregression;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.classification.linearsvc.LinearSVC.ParseSample;
import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.common.lossfunc.BinaryLogisticLoss;
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
 * An Estimator which implements the logistic regression algorithm.
 *
 * <p>See https://en.wikipedia.org/wiki/Logistic_regression.
 */
public class LogisticRegression
        implements Estimator<LogisticRegression, LogisticRegressionModel>,
                LogisticRegressionParams<LogisticRegression> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public LogisticRegression() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    @SuppressWarnings({"rawTypes", "ConstantConditions"})
    public LogisticRegressionModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        String classificationType = getMultiClass();
        Preconditions.checkArgument(
                "auto".equals(classificationType) || "binomial".equals(classificationType),
                "Multinomial classification is not supported yet. Supported options: [auto, binomial].");
        String dataName = "data";
        String modelName = "modelName";
        MLData mlData = MLData.of(inputs, new String[] {dataName});
        // AlgorithmFlow algoFlow = new AlgorithmFlow();

        // algoFlow.add(new ParseSample(getLabelCol(), getWeightCol(), getFeaturesCol()))
        //        .add(
        //                new MapComponent<LabeledPointWithWeight, Integer>(
        //                                x -> x.getFeatures().size())
        //                        .output(modelName))
        //        .add(
        //                new ReduceComponent<Integer>() {
        //                    @Override
        //                    public Integer reduce(Integer t0, Integer t1) {
        //                        Preconditions.checkState(
        //                                t0.equals(t1),
        //                                "The training data should all have same dimensions.");
        //                        return t0;
        //                    }
        //                })
        //        .add(new MapComponent<Integer, DenseVector>(DenseVector::new))
        //        .add(
        //                new OptimizerComponent(paramMap, Method.SGD, BinaryLogisticLoss.INSTANCE)
        //                        .withInitModel(modelName)
        //                        .input(dataName)
        //                        .output(modelName))
        //        .add(
        //                new MapComponent<DenseVector, LogisticRegressionModelData>() {
        //                    @Override
        //                    public LogisticRegressionModelData map(DenseVector denseVector) {
        //                        return new LogisticRegressionModelData(denseVector, 0L);
        //                    }
        //                });

        mlData.map(new ParseSample(getLabelCol(), getWeightCol(), getFeaturesCol()));
        mlData.map(
                null,
                modelName,
                (MapFunction<LabeledPointWithWeight, Integer>) (x -> x.getFeatures().size()));
        new MLDataFunction(
                        "reduce",
                        (ReduceFunction<Integer>)
                                (t0, t1) -> {
                                    Preconditions.checkState(
                                            t0.equals(t1),
                                            "The training data should all have same dimensions.");
                                    return t0;
                                })
                .apply(mlData);
        mlData.map((MapFunction<Integer, DenseVector>) (DenseVector::new));
        new OptimizerComponent(paramMap, Method.SGD, BinaryLogisticLoss.INSTANCE)
                .withInitModel(modelName)
                .input(dataName)
                .output(modelName)
                .apply(mlData);
        mlData.map(
                (MapFunction<DenseVector, LogisticRegressionModelData>)
                        denseVector -> new LogisticRegressionModelData(denseVector, 0L));

        LogisticRegressionModel model =
                new LogisticRegressionModel().setModelData(mlData.getTable(modelName));
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static LogisticRegression load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
