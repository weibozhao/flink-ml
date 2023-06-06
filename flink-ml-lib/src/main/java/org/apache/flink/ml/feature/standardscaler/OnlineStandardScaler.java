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

package org.apache.flink.ml.feature.standardscaler;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.window.EventTimeSessionWindows;
import org.apache.flink.ml.common.window.EventTimeTumblingWindows;
import org.apache.flink.ml.common.window.Windows;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.functions.windowing.ProcessAllWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.Window;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * An Estimator which implements the online standard scaling algorithm, which is the online version
 * of {@link StandardScaler}.
 *
 * <p>OnlineStandardScaler splits the input data by the user-specified window strategy (i.e., {@link
 * org.apache.flink.ml.common.param.HasWindows}). For each window, it computes the mean and standard
 * deviation using the data seen so far (i.e., not only the data in the current window, but also the
 * history data). The model data generated by OnlineStandardScaler is a model stream. There is one
 * model data for each window.
 *
 * <p>During the inference phase (i.e., using {@link OnlineStandardScalerModel} for prediction),
 * users could output the model version that is used for predicting each data point. Moreover,
 *
 * <ul>
 *   <li>When the train data and test data both contain event time, users could specify the maximum
 *       difference between the timestamps of the input and model data ({@link
 *       org.apache.flink.ml.common.param.HasMaxAllowedModelDelayMs}), which enforces to use a
 *       relatively fresh model for prediction.
 *   <li>Otherwise, the prediction process always uses the current model data for prediction.
 * </ul>
 */
public class OnlineStandardScaler
        implements Estimator<OnlineStandardScaler, OnlineStandardScalerModel>,
                OnlineStandardScalerParams<OnlineStandardScaler> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public OnlineStandardScaler() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public OnlineStandardScalerModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        Windows windows = getWindows();

        boolean isEventTimeBasedTraining = false;
        if (windows instanceof EventTimeTumblingWindows
                || windows instanceof EventTimeSessionWindows) {
            isEventTimeBasedTraining = true;
        }

        DataStream<StandardScalerModelData> modelData =
                DataStreamUtils.windowAllAndProcess(
                        tEnv.toDataStream(inputs[0]),
                        windows,
                        new ComputeModelDataFunction<>(getInputCol(), isEventTimeBasedTraining));

        OnlineStandardScalerModel model =
                new OnlineStandardScalerModel().setModelData(tEnv.fromDataStream(modelData));
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    private static class ComputeModelDataFunction<W extends Window>
            extends ProcessAllWindowFunction<Row, StandardScalerModelData, W> {

        private final String inputCol;
        private final boolean isEventTimeBasedTraining;

        public ComputeModelDataFunction(String inputCol, boolean isEventTimeBasedTraining) {
            this.inputCol = inputCol;
            this.isEventTimeBasedTraining = isEventTimeBasedTraining;
        }

        @Override
        public void process(
                ProcessAllWindowFunction<Row, StandardScalerModelData, W>.Context context,
                Iterable<Row> iterable,
                Collector<StandardScalerModelData> collector)
                throws Exception {
            ListState<DenseIntDoubleVector> sumState =
                    context.globalState()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "sumState", DenseIntDoubleVectorTypeInfo.INSTANCE));
            ListState<DenseIntDoubleVector> squaredSumState =
                    context.globalState()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "squaredSumState",
                                            DenseIntDoubleVectorTypeInfo.INSTANCE));
            ListState<Long> numElementsState =
                    context.globalState()
                            .getListState(
                                    new ListStateDescriptor<>("numElementsState", Types.LONG));
            ListState<Long> modelVersionState =
                    context.globalState()
                            .getListState(
                                    new ListStateDescriptor<>("modelVersionState", Types.LONG));
            DenseIntDoubleVector sum =
                    OperatorStateUtils.getUniqueElement(sumState, "sumState").orElse(null);
            DenseIntDoubleVector squaredSum =
                    OperatorStateUtils.getUniqueElement(squaredSumState, "squaredSumState")
                            .orElse(null);
            long numElements =
                    OperatorStateUtils.getUniqueElement(numElementsState, "numElementsState")
                            .orElse(0L);
            long modelVersion =
                    OperatorStateUtils.getUniqueElement(modelVersionState, "modelVersionState")
                            .orElse(0L);

            long numElementsBefore = numElements;
            for (Row element : iterable) {
                IntDoubleVector inputVec =
                        ((IntDoubleVector) Objects.requireNonNull(element.getField(inputCol)))
                                .clone();
                if (numElements == 0) {
                    sum = new DenseIntDoubleVector(inputVec.size());
                    squaredSum = new DenseIntDoubleVector(inputVec.size());
                }
                BLAS.axpy(1, inputVec, sum);
                BLAS.hDot(inputVec, inputVec);
                BLAS.axpy(1, inputVec, squaredSum);
                numElements++;
            }
            if (numElements - numElementsBefore > 0) {
                long currentEventTime =
                        isEventTimeBasedTraining ? context.window().maxTimestamp() : Long.MAX_VALUE;
                collector.collect(
                        buildModelData(
                                numElements,
                                sum.clone(),
                                squaredSum.clone(),
                                modelVersion,
                                currentEventTime));

                sumState.update(Collections.singletonList(sum));
                squaredSumState.update(Collections.singletonList(squaredSum));
                numElementsState.update(Collections.singletonList(numElements));
                modelVersion++;
                modelVersionState.update(Collections.singletonList(modelVersion));
            }
        }
    }

    private static StandardScalerModelData buildModelData(
            long numElements,
            DenseIntDoubleVector sum,
            DenseIntDoubleVector squaredSum,
            long modelVersion,
            long currentTimeStamp) {
        BLAS.scal(1.0 / numElements, sum);
        double[] mean = sum.values;
        double[] std = squaredSum.values;
        if (numElements > 1) {
            for (int i = 0; i < mean.length; i++) {
                std[i] =
                        Math.sqrt(
                                (squaredSum.values[i] - numElements * mean[i] * mean[i])
                                        / (numElements - 1));
            }
        } else {
            Arrays.fill(std, 0.0);
        }

        return new StandardScalerModelData(
                Vectors.dense(mean), Vectors.dense(std), modelVersion, currentTimeStamp);
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    public static OnlineStandardScaler load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }
}
