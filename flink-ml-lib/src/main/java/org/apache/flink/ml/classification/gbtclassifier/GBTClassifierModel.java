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

package org.apache.flink.ml.classification.gbtclassifier;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.gbt.BaseGBTModel;
import org.apache.flink.ml.common.gbt.GBTModelData;
import org.apache.flink.ml.common.gbt.GBTModelDataUtil;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;
import org.eclipse.collections.impl.map.mutable.primitive.IntDoubleHashMap;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;

/** A Model computed by {@link GBTClassifier}. */
public class GBTClassifierModel extends BaseGBTModel<GBTClassifierModel>
        implements GBTClassifierModelParams<GBTClassifierModel> {

    /**
     * Loads model data from path.
     *
     * @param tEnv A StreamTableEnvironment instance.
     * @param path Model path.
     * @return GBT classification model.
     */
    public static GBTClassifierModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return BaseGBTModel.load(tEnv, path);
    }

    public static GBTClassifierModelServable loadServable(String path) throws IOException {
        return GBTClassifierModelServable.load(path);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> inputStream = tEnv.toDataStream(inputs[0]);
        final String broadcastModelKey = "broadcastModelKey";
        DataStream<GBTModelData> modelDataStream =
                GBTModelDataUtil.getModelDataStream(modelDataTable);
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(),
                                Types.DOUBLE,
                                DenseVectorTypeInfo.INSTANCE,
                                DenseVectorTypeInfo.INSTANCE),
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldNames(),
                                getPredictionCol(),
                                getRawPredictionCol(),
                                getProbabilityCol()));
        DataStream<Row> predictionResult =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(inputStream),
                        Collections.singletonMap(broadcastModelKey, modelDataStream),
                        inputList -> {
                            //noinspection unchecked
                            DataStream<Row> inputData = (DataStream<Row>) inputList.get(0);
                            return inputData.map(
                                    new PredictLabelFunction(
                                            broadcastModelKey, getFeaturesCols(), paramMap),
                                    outputTypeInfo);
                        });
        return new Table[] {tEnv.fromDataStream(predictionResult)};
    }

    private static class PredictLabelFunction extends RichMapFunction<Row, Row> {

        private final String broadcastModelKey;
        private final Map<Param<?>, Object> params;
        private String[] featuresCols;
        private GBTModelData modelData;

        private GBTClassifierModelServable servable;

        public PredictLabelFunction(
                String broadcastModelKey, String[] featuresCols, Map<Param<?>, Object> paramMap) {
            this.broadcastModelKey = broadcastModelKey;
            this.featuresCols = featuresCols;
            params = paramMap;
        }

        @Override
        public Row map(Row value) {
            if (null == servable) {
                modelData =
                        (GBTModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastModelKey).get(0);
                servable = new GBTClassifierModelServable(modelData);
                ParamUtils.updateExistingParams(servable, params);
                // TODO: this is a temporary approach to pass parameter values from training to
                // prediction for PAI Designer.
                if (null == featuresCols || 0 == featuresCols.length) {
                    featuresCols = modelData.featureNames.toArray(new String[0]);
                    if (modelData.isInputVector) {
                        String originalVectorCol =
                                featuresCols[0].substring(1, featuresCols[0].lastIndexOf("_f"));
                        featuresCols = new String[] {originalVectorCol};
                    }
                }
            }
            IntDoubleHashMap features = modelData.toFeatures(featuresCols, value::getField);
            Tuple3<Double, DenseVector, DenseVector> results = servable.transform(features);
            return Row.join(value, Row.of(results.f0, results.f1, results.f2));
        }
    }
}
