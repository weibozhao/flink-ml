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

package org.apache.flink.ml.clustering.kmeans;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.clustering.kmeans.KMeansModelDataUtil.ModelDataDecoder;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.distance.DistanceMeasure;
import org.apache.flink.ml.common.ps.api.MLData;
import org.apache.flink.ml.common.ps.api.MLData.MLDataFunction;
import org.apache.flink.ml.common.ps.api.ModelParseComponent;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.VectorWithNorm;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/** A Model which clusters data into k clusters using the model data computed by {@link KMeans}. */
public class KMeansModel implements Model<KMeansModel>, KMeansModelParams<KMeansModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public KMeansModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public KMeansModel setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        MLData mlData =
                MLData.of(new Table[] {inputs[0], modelDataTable}, new String[] {"data", "model"});

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), Types.INT),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getPredictionCol()));

        mlData = new ModelParseComponent<>("model", KMeansModelData.class).apply(mlData);
        mlData =
                new MLDataFunction(
                                "map",
                                new PredictComponent(
                                        getFeaturesCol(),
                                        DistanceMeasure.getInstance(getDistanceMeasure()),
                                        getK(),
                                        "model"))
                        .withBroadcast("model")
                        .returns(outputTypeInfo)
                        .apply(mlData);

        return mlData.slice("data").getTables();
    }

    /** A utility function used for prediction. */
    private static class PredictComponent extends RichMapFunction<Row, Row> {

        private final String featuresCol;
        private String broadcastName;

        private final DistanceMeasure distanceMeasure;

        private final int k;

        private VectorWithNorm[] centroids;

        public PredictComponent(
                String featuresCol, DistanceMeasure distanceMeasure, int k, String broadcastName) {
            this.featuresCol = featuresCol;
            this.distanceMeasure = distanceMeasure;
            this.k = k;
            this.broadcastName = broadcastName;
        }

        @Override
        public Row map(Row dataPoint) {
            if (centroids == null) {
                KMeansModelData modelData =
                        (KMeansModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastName).get(0);
                Preconditions.checkArgument(modelData.centroids.length <= k);
                centroids = new VectorWithNorm[modelData.centroids.length];
                for (int i = 0; i < modelData.centroids.length; i++) {
                    centroids[i] = new VectorWithNorm(modelData.centroids[i]);
                }
            }
            DenseVector point = ((Vector) dataPoint.getField(featuresCol)).toDense();
            int closestCentroidId =
                    distanceMeasure.findClosest(centroids, new VectorWithNorm(point));
            return Row.join(dataPoint, Row.of(closestCentroidId));
        }
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveModelData(
                KMeansModelDataUtil.getModelDataStream(modelDataTable),
                path,
                new KMeansModelDataUtil.ModelDataEncoder());
        ReadWriteUtils.saveMetadata(this, path);
    }

    // TODO: Add INFO level logging.
    public static KMeansModel load(StreamTableEnvironment tEnv, String path) throws IOException {
        Table modelDataTable = ReadWriteUtils.loadModelData(tEnv, path, new ModelDataDecoder());
        KMeansModel model = ReadWriteUtils.loadStageParam(path);
        return model.setModelData(modelDataTable);
    }
}
