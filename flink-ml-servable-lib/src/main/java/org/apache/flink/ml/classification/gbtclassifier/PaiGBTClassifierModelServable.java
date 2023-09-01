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

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.gbt.GBTModelData;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.servable.api.DataFrame;
import org.apache.flink.ml.servable.api.PaiModelServable;
import org.apache.flink.ml.servable.api.Row;
import org.apache.flink.ml.servable.types.DataType;
import org.apache.flink.ml.servable.types.DataTypes;
import org.apache.flink.ml.util.ServableReadWriteUtils;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.apache.flink.ml.util.FileUtils.loadMetadata;

/** A Servable which can be used to classifies data in online inference. */
@Deprecated
public class PaiGBTClassifierModelServable extends GBTClassifierModelServable
        implements PaiModelServable<PaiGBTClassifierModelServable> {
    private String predictionDetailCol;

    public PaiGBTClassifierModelServable() {
        super();
    }

    PaiGBTClassifierModelServable(GBTModelData modelData) {
        super(modelData);
    }

    @Override
    public Tuple2<String[], DataType[]> getResultNamesAndTypes() {
        return Tuple2.of(
                new String[] {getPredictionCol(), predictionDetailCol},
                new DataType[] {DataTypes.DOUBLE, DataTypes.STRING});
    }

    @Override
    public PaiGBTClassifierModelServable setModelData(List<Row> modelRows) throws IOException {
        modelRows.sort(Comparator.comparing(d -> d.<Integer>getAs(0)));
        String b64str =
                modelRows.stream().map(d -> d.<String>getAs(1)).collect(Collectors.joining());
        byte[] bytes = Base64.getDecoder().decode(b64str);
        setModelData(new ByteArrayInputStream(bytes));
        return this;
    }

    @Override
    public void setDesignerParams(Map<String, Object> designerParams) {
        if (designerParams.containsKey("predictionCol")) {
            setPredictionCol((String) designerParams.get("predictionCol"));
        }
        predictionDetailCol = (String) designerParams.get("predictionDetailCol");
    }

    @Override
    public DataFrame transform(DataFrame input) {
        DataFrame output = super.transform(input);
        int probabilityColIndex = input.getIndex(getProbabilityCol());
        List<String> predictDetails = new ArrayList<>();
        final String template = "{\"0\": %f, \"1\": %f}";
        for (Row row : output.collect()) {
            Vector probabilities = row.getAs(probabilityColIndex);
            predictDetails.add(String.format(template, probabilities.get(0), probabilities.get(1)));
        }
        output.addColumn(predictionDetailCol, DataTypes.STRING, predictDetails);
        return output;
    }

    public static PaiGBTClassifierModelServable load(String path) throws IOException {
        Map<String, Object> designerParams = (Map<String, Object>) loadMetadata(path, "");
        PaiGBTClassifierModelServable paiServable = new PaiGBTClassifierModelServable();
        paiServable.setDesignerParams(designerParams);
        Path tmpPath = Paths.get(path);
        Path mergePath = tmpPath.resolve(MODEL_DATA_PATH);
        try (InputStream modelData = ServableReadWriteUtils.loadModelData(mergePath.toString())) {
            paiServable.setModelData(modelData);
            return paiServable;
        }
    }
}
