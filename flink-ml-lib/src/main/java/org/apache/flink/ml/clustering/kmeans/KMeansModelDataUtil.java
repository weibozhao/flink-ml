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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Random;

/**
 * Model data of {@link KMeansModel} and {@link OnlineKMeansModel}.
 *
 * <p>This class also provides methods to convert model data from Table to Datastream, and classes
 * to save/load model data.
 */
public class KMeansModelDataUtil {

    /**
     * Generates a Table containing a {@link KMeansModelDataUtil} instance with randomly generated
     * centroids.
     *
     * @param tEnv The environment where to create the table.
     * @param k The number of generated centroids.
     * @param dim The size of generated centroids.
     * @param weight The weight of the centroids.
     * @param seed Random seed.
     */
    public static Table generateRandomModelData(
            StreamTableEnvironment tEnv, int k, int dim, double weight, long seed) {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);
        return tEnv.fromDataStream(
                env.fromElements(1).map(new RandomCentroidsCreator(k, dim, weight, seed)));
    }

    private static class RandomCentroidsCreator implements MapFunction<Integer, KMeansModelData> {
        private final int k;
        private final int dim;
        private final double weight;
        private final long seed;

        private RandomCentroidsCreator(int k, int dim, double weight, long seed) {
            this.k = k;
            this.dim = dim;
            this.weight = weight;
            this.seed = seed;
        }

        @Override
        public KMeansModelData map(Integer integer) {
            DenseVector[] centroids = new DenseVector[k];
            Random random = new Random(seed);
            for (int i = 0; i < k; i++) {
                centroids[i] = new DenseVector(dim);
                for (int j = 0; j < dim; j++) {
                    centroids[i].values[j] = random.nextDouble();
                }
            }
            DenseVector weights = new DenseVector(k);
            Arrays.fill(weights.values, weight);
            return new KMeansModelData(centroids, weights);
        }
    }

    /**
     * Converts the table model to a data stream.
     *
     * @param modelData The table model data.
     * @return The data stream model data.
     */
    public static DataStream<KMeansModelData> getModelDataStream(Table modelData) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelData).getTableEnvironment();
        return tEnv.toDataStream(modelData)
                .map(
                        x ->
                                new KMeansModelData(
                                        Arrays.stream(((Vector[]) x.getField(0)))
                                                .map(Vector::toDense)
                                                .toArray(DenseVector[]::new),
                                        ((Vector) x.getField(1)).toDense()));
    }

    /** Data encoder for {@link KMeansModelDataUtil}. */
    public static class ModelDataEncoder implements Encoder<KMeansModelData> {

        @Override
        public void encode(KMeansModelData modelData, OutputStream outputStream)
                throws IOException {
            modelData.encode(outputStream);
        }
    }

    /** Data decoder for {@link KMeansModelDataUtil}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<KMeansModelData> {
        @Override
        public Reader<KMeansModelData> createReader(
                Configuration configuration, FSDataInputStream inputStream) {
            return new Reader<KMeansModelData>() {

                @Override
                public KMeansModelData read() throws IOException {
                    return KMeansModelData.decode(inputStream);
                }

                @Override
                public void close() throws IOException {
                    inputStream.close();
                }
            };
        }

        @Override
        public TypeInformation<KMeansModelData> getProducedType() {
            return TypeInformation.of(KMeansModelData.class);
        }
    }
}
