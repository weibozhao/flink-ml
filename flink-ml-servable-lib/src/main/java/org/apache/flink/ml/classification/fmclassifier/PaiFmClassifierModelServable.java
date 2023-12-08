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

package org.apache.flink.ml.classification.fmclassifier;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.fm.FmModelData;
import org.apache.flink.ml.common.fm.FmModelServable;
import org.apache.flink.ml.servable.api.PaiModelServable;
import org.apache.flink.ml.servable.api.Row;
import org.apache.flink.ml.servable.types.BasicType;
import org.apache.flink.ml.servable.types.DataType;
import org.apache.flink.ml.servable.types.DataTypes;
import org.apache.flink.ml.util.ServableReadWriteUtils;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Base64;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.apache.flink.ml.util.FileUtils.loadMetadata;

/** A Servable which can be used to classifies data with fm model in online inference for Pai. */
@Deprecated
public class PaiFmClassifierModelServable extends FmModelServable
        implements PaiModelServable<PaiFmClassifierModelServable> {

    public PaiFmClassifierModelServable() {
        super();
    }

    PaiFmClassifierModelServable(List<FmModelData> modelData) {
        super(modelData);
    }

    @Override
    public Tuple2<String[], DataType[]> getResultNamesAndTypes() {
        return Tuple2.of(
                new String[] {getPredictionCol(), getRawPredictionCol()},
                new DataType[] {DataTypes.DOUBLE, DataTypes.vectorType(BasicType.DOUBLE)});
    }

    @Override
    public PaiFmClassifierModelServable setModelData(List<Row> modelRows) throws IOException {
        modelRows.sort(Comparator.comparing(d -> d.<Integer>getAs(0)));
        String b64str =
                modelRows.stream().map(d -> d.<String>getAs(1)).collect(Collectors.joining());
        byte[] bytes = Base64.getDecoder().decode(b64str);
        setModelData(new ByteArrayInputStream(bytes));
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
        if (mapDesignerParams.containsKey("predictionDetailCol")) {
            setRawPredictionCol((String) mapDesignerParams.get("predictionDetailCol"));
        }
    }

    public static PaiFmClassifierModelServable load(String path) throws IOException {
        Map<String, Object> designerParams = (Map<String, Object>) loadMetadata(path, "");
        PaiFmClassifierModelServable paiServable = new PaiFmClassifierModelServable();
        paiServable.setDesignerParams(designerParams);
        try (InputStream modelData = ServableReadWriteUtils.loadModelData(path)) {
            paiServable.setModelData(modelData);
            return paiServable;
        }
    }
}
