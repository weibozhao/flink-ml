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

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.fm.FmModelData;
import org.apache.flink.ml.common.fm.FmModelDataUtil;
import org.apache.flink.ml.common.fm.FmModelServable;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** A Model which classifies data using the model data computed by {@link FmRegressor}. */
public class FmRegressorModel
        implements Model<FmRegressorModel>, FmRegressorModelParams<FmRegressorModel> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    private Table modelDataTable;

    public FmRegressorModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    @SuppressWarnings("unchecked")
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> inputStream = tEnv.toDataStream(inputs[0]);
        final String broadcastModelKey = "broadcastModelKey";
        DataStream<FmModelData> modelDataStream =
                FmModelDataUtil.getModelDataStream(modelDataTable);
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(), BasicTypeInfo.DOUBLE_TYPE_INFO),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getPredictionCol()));
        DataStream<Row> predictionResult =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(inputStream),
                        Collections.singletonMap(broadcastModelKey, modelDataStream),
                        inputList -> {
                            DataStream<Row> inputData = (DataStream<Row>) inputList.get(0);
                            return inputData.map(
                                    new PredictLabelFunction(broadcastModelKey, paramMap),
                                    outputTypeInfo);
                        });
        return new Table[] {tEnv.fromDataStream(predictionResult)};
    }

    @Override
    public FmRegressorModel setModelData(Table... inputs) {
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                FmModelDataUtil.getModelDataStream(modelDataTable),
                path,
                new FmModelDataUtil.ModelDataEncoder());
    }

    public static FmRegressorModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        FmRegressorModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(tEnv, path, new FmModelDataUtil.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    public static FmModelServable loadServable(String path) throws IOException {
        return FmModelServable.load(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /** A utility function used for prediction. */
    private static class PredictLabelFunction extends RichMapFunction<Row, Row> {

        private final String broadcastModelKey;

        private final Map<Param<?>, Object> params;

        private FmModelServable servable;

        public PredictLabelFunction(String broadcastModelKey, Map<Param<?>, Object> params) {
            this.broadcastModelKey = broadcastModelKey;
            this.params = params;
        }

        @Override
        public Row map(Row dataPoint) {
            if (servable == null) {
                List<FmModelData> modelData =
                        getRuntimeContext().getBroadcastVariable(broadcastModelKey);

                servable = new FmModelServable(modelData);

                ParamUtils.updateExistingParams(servable, params);
            }
            SparseVector features = (SparseVector) dataPoint.getField(servable.getFeaturesCol());

            Tuple2<Double, DenseVector> predictionResult = servable.transform(features);

            return Row.join(dataPoint, Row.of(predictionResult.f0));
        }
    }
}
