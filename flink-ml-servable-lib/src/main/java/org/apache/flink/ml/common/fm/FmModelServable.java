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

package org.apache.flink.ml.common.fm;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.classification.fmclassifier.FmClassifierModelParams;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.servable.api.DataFrame;
import org.apache.flink.ml.servable.api.ModelServable;
import org.apache.flink.ml.servable.api.Row;
import org.apache.flink.ml.servable.types.BasicType;
import org.apache.flink.ml.servable.types.DataTypes;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ServableReadWriteUtils;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** A Servable which can be used fm classification or regression transform in online inference. */
public class FmModelServable
        implements ModelServable<FmModelServable>, FmClassifierModelParams<FmModelServable> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    private final Map<Long, float[]> factors = new HashMap<>();
    private int[] dim;
    private int k;
    private float[] zeros;
    private boolean isReg;

    public FmModelServable() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    public FmModelServable(List<FmModelData> modelList) {
        this();
        for (FmModelData fmModelData : modelList) {
            for (Tuple2<Long, float[]> t2 : fmModelData.factors) {
                this.factors.put(t2.f0, t2.f1);
            }
            this.dim = fmModelData.dim;
            this.k = dim[2];
            this.zeros = new float[k + 1];
            this.isReg = fmModelData.isReg;
        }
    }

    @Override
    public DataFrame transform(DataFrame input) {

        List<Double> predictionResults = new ArrayList<>();
        List<DenseVector> rawPredictionResults = new ArrayList<>();

        int featuresColIndex = input.getIndex(getFeaturesCol());
        for (Row row : input.collect()) {
            SparseVector features = (SparseVector) row.get(featuresColIndex);
            Tuple2<Double, DenseVector> dataPoint = transform(features);
            predictionResults.add(dataPoint.f0);
            if (!isReg) {
                rawPredictionResults.add(dataPoint.f1);
            }
        }

        input.addColumn(getPredictionCol(), DataTypes.DOUBLE, predictionResults);
        if (!isReg) {
            input.addColumn(
                    getRawPredictionCol(),
                    DataTypes.VECTOR(BasicType.DOUBLE),
                    rawPredictionResults);
        }
        return input;
    }

    public FmModelServable setModelData(InputStream... modelDataInputs) throws IOException {
        Preconditions.checkArgument(modelDataInputs.length == 1);
        FmModelData fmModelData = FmModelData.decode(modelDataInputs[0]);
        for (Tuple2<Long, float[]> t2 : fmModelData.factors) {
            this.factors.put(t2.f0, t2.f1);
        }

        this.dim = fmModelData.dim;
        this.k = dim[2];
        this.zeros = new float[k + 1];
        this.isReg = fmModelData.isReg;
        return this;
    }

    public static FmModelServable load(String path) throws IOException {
        FmModelServable servable =
                ServableReadWriteUtils.loadServableParam(path, FmModelServable.class);

        try (InputStream modelData = ServableReadWriteUtils.loadModelData(path)) {
            servable.setModelData(modelData);
            return servable;
        }
    }

    /**
     * The main logic that predicts one input data point.
     *
     * @param feature The input feature.
     * @return The prediction label and the raw probabilities.
     */
    public Tuple2<Double, DenseVector> transform(SparseVector feature) {
        double y = predict(feature);
        if (this.isReg) {
            return Tuple2.of(y, Vectors.dense(y));
        } else {
            double prob = 1. / (1. + Math.exp(-y));
            return Tuple2.of(y >= 0 ? 1. : 0., Vectors.dense(1 - prob, prob));
        }
    }

    public double predict(SparseVector sample) {
        int[] featureIds = sample.indices;
        double[] featureValues = sample.values;

        float[] vx = new float[k];
        float[] v2x2 = new float[k];

        float y = 0.0f;
        final long biasPos = -1L;

        // the bias term
        if (dim[0] > 0) {
            float[] w = factors.get(biasPos);
            y += w[0];
        }

        for (int i = 0; i < featureIds.length; i++) {
            long featureId = featureIds[i];
            double x = featureValues[i];
            float[] weights = factors.get(featureId);
            float[] w = weights != null ? weights : zeros;
            if (dim[1] > 0) {
                // the linear term, w's last pos is reserved for the linear weights.
                y += x * w[k];
            }

            // the quadratic term
            for (int j = 0; j < k; j++) {
                double viXi = x * w[j];
                vx[j] += viXi;
                v2x2[j] += viXi * viXi;
            }
        }

        for (int i = 0; i < k; i++) {
            y += 0.5 * (vx[i] * vx[i] - v2x2[i]);
        }

        return y;
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
