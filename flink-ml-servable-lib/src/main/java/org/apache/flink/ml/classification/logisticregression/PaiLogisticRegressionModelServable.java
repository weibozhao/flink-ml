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

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorSerializer;
import org.apache.flink.ml.servable.api.PaiModelServable;
import org.apache.flink.ml.servable.api.Row;
import org.apache.flink.ml.servable.types.BasicType;
import org.apache.flink.ml.servable.types.DataType;
import org.apache.flink.ml.servable.types.DataTypes;
import org.apache.flink.ml.util.ServableReadWriteUtils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;

import static org.apache.flink.ml.util.FileUtils.loadMetadata;

/**
 * A Servable which can be used to classifies data in online inference for Pai. This is just a demo
 * to show the use of {@link PaiModelServable} interfaces.
 */
@Deprecated
public class PaiLogisticRegressionModelServable extends LogisticRegressionModelServable
        implements PaiModelServable<PaiLogisticRegressionModelServable> {

    public PaiLogisticRegressionModelServable() {
        super();
    }

    PaiLogisticRegressionModelServable(LogisticRegressionModelData modelData) {
        super(modelData);
    }

    @Override
    public Tuple2<String[], DataType[]> getResultNamesAndTypes() {
        return Tuple2.of(
                new String[] {getPredictionCol(), getRawPredictionCol()},
                new DataType[] {DataTypes.DOUBLE, DataTypes.vectorType(BasicType.DOUBLE)});
    }

    @Override
    public PaiLogisticRegressionModelServable setModelData(List<Row> modelRows) throws IOException {
        DenseVector vec = (DenseVector) modelRows.get(0).get(0);
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        DataOutputViewStreamWrapper dataOutputViewStreamWrapper =
                new DataOutputViewStreamWrapper(outputStream);

        DenseVectorSerializer serializer = new DenseVectorSerializer();
        serializer.serialize(vec, dataOutputViewStreamWrapper);
        dataOutputViewStreamWrapper.writeLong(1L);

        setModelData(new ByteArrayInputStream(outputStream.toByteArray()));
        return this;
    }

    @Override
    public void setDesignerParams(Map<String, Object> mapDesignerParams) {
        if (mapDesignerParams.containsKey("featuresCol")) {
            setFeaturesCol((String) mapDesignerParams.get("featuresCol"));
        }
        if (mapDesignerParams.containsKey("predictionCol")) {
            setPredictionCol((String) mapDesignerParams.get("predictionCol"));
        }
        if (mapDesignerParams.containsKey("rawPredictionCol")) {
            setRawPredictionCol((String) mapDesignerParams.get("rawPredictionCol"));
        }
    }

    public static PaiLogisticRegressionModelServable load(String path) throws IOException {
        Map<String, Object> designerParams = (Map<String, Object>) loadMetadata(path, "");
        PaiLogisticRegressionModelServable paiServable = new PaiLogisticRegressionModelServable();
        paiServable.setDesignerParams(designerParams);
        try (InputStream modelData = ServableReadWriteUtils.loadModelData(path)) {
            paiServable.setModelData(modelData);
            return paiServable;
        }
    }
}
