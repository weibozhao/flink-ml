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

import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorSerializer;
import org.apache.flink.table.api.Table;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Model data of {@link KMeansModel} and {@link OnlineKMeansModel}.
 *
 * <p>This class also provides methods to convert model data from Table to Datastream, and classes
 * to save/load model data.
 */
public class KMeansModelData {

    public DenseVector[] centroids;

    /**
     * The weight of the centroids. It is used when updating the model data in online training
     * process.
     *
     * <p>KMeansModelData objects generated during {@link KMeans#fit(Table...)} also contains this
     * field, so that it can be used as the initial model data of the online training process.
     */
    public DenseVector weights;

    public KMeansModelData(DenseVector[] centroids, DenseVector weights) {
        Preconditions.checkArgument(centroids.length == weights.size());
        this.centroids = centroids;
        this.weights = weights;
    }

    public KMeansModelData(Row row) {
        this.centroids = row.getFieldAs(0);
        this.weights = row.getFieldAs(1);
    }

    public KMeansModelData() {}

    public void encode(OutputStream outputStream) throws IOException {
        final DenseVectorSerializer serializer = new DenseVectorSerializer();

        DataOutputViewStreamWrapper outputViewStreamWrapper =
                new DataOutputViewStreamWrapper(outputStream);
        IntSerializer.INSTANCE.serialize(centroids.length, outputViewStreamWrapper);
        for (DenseVector denseVector : centroids) {
            serializer.serialize(denseVector, new DataOutputViewStreamWrapper(outputStream));
        }
        serializer.serialize(weights, new DataOutputViewStreamWrapper(outputStream));
    }

    public static KMeansModelData decode(InputStream inputStream) throws IOException {
        final DenseVectorSerializer serializer = new DenseVectorSerializer();
        try {
            DataInputViewStreamWrapper inputViewStreamWrapper =
                    new DataInputViewStreamWrapper(inputStream);
            int numDenseVectors = IntSerializer.INSTANCE.deserialize(inputViewStreamWrapper);
            DenseVector[] centroids = new DenseVector[numDenseVectors];
            for (int i = 0; i < numDenseVectors; i++) {
                centroids[i] = serializer.deserialize(inputViewStreamWrapper);
            }
            DenseVector weights = serializer.deserialize(inputViewStreamWrapper);
            return new KMeansModelData(centroids, weights);
        } catch (EOFException e) {
            return null;
        }
    }
}
