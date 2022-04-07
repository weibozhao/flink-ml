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

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.classification.ftrl.FtrlModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorSerializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Model data of {@link LogisticRegressionModel}, {@link FtrlModel}.
 *
 * <p>This class also provides methods to convert model data from Table to Datastream, and classes
 * to save/load model data.
 */
public class LinearModelData {

    public DenseVector coefficient;
    public long modelVersion;

    public LinearModelData(DenseVector coefficient, long modelVersion) {
        this.coefficient = coefficient;
        this.modelVersion = modelVersion;
    }

    public LinearModelData() {}

    /**
     * Converts the table model to a data stream.
     *
     * @param modelData The table model data.
     * @return The data stream model data.
     */
    public static DataStream<LinearModelData> getModelDataStream(Table modelData) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelData).getTableEnvironment();
        return tEnv.toDataStream(modelData)
                .map(x -> new LinearModelData((DenseVector) x.getField(0), x.getFieldAs(1)));
    }

    /** Data encoder for {@link LogisticRegressionModel}. */
    public static class ModelDataEncoder implements Encoder<LinearModelData> {

        @Override
        public void encode(LinearModelData modelData, OutputStream outputStream)
                throws IOException {
            DataOutputViewStreamWrapper dataOutputViewStreamWrapper = new DataOutputViewStreamWrapper(outputStream);
            DenseVectorSerializer.INSTANCE.serialize(
                    modelData.coefficient, dataOutputViewStreamWrapper);
            dataOutputViewStreamWrapper.writeLong(modelData.modelVersion);
        }
    }

    /** Data decoder for {@link LogisticRegressionModel}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<LinearModelData> {

        @Override
        public Reader<LinearModelData> createReader(
                Configuration configuration, FSDataInputStream inputStream) {
            return new Reader<LinearModelData>() {

                @Override
                public LinearModelData read() throws IOException {
                    try {
                        DataInputViewStreamWrapper dataInputViewStreamWrapper = new DataInputViewStreamWrapper(inputStream);
                        DenseVector coefficient =
                                DenseVectorSerializer.INSTANCE.deserialize(dataInputViewStreamWrapper);
                        long modelVersion = dataInputViewStreamWrapper.readLong();
                        return new LinearModelData(coefficient, modelVersion);
                    } catch (EOFException e) {
                        return null;
                    }
                }

                @Override
                public void close() throws IOException {
                    inputStream.close();
                }
            };
        }

        @Override
        public TypeInformation<LinearModelData> getProducedType() {
            return TypeInformation.of(LinearModelData.class);
        }
    }
}
