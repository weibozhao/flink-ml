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

package org.apache.flink.ml.feature.idf;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.base.LongSerializer;
import org.apache.flink.api.common.typeutils.base.array.LongPrimitiveArraySerializer;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorSerializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;

import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Model data of {@link IDFModel}.
 *
 * <p>This class also provides methods to convert model data from Table to a data stream, and
 * classes to save/load model data.
 */
public class IDFModelData {
    /** Inverse document frequency for all terms. */
    public DenseIntDoubleVector idf;
    /** Document frequency for all terms after filtering out infrequent terms. */
    public long[] docFreq;
    /** Number of docs in the training set. */
    public long numDocs;

    public IDFModelData() {}

    public IDFModelData(DenseIntDoubleVector idf, long[] docFreq, long numDocs) {
        this.idf = idf;
        this.docFreq = docFreq;
        this.numDocs = numDocs;
    }

    /**
     * Converts the table model to a data stream.
     *
     * @param modelDataTable The table model data.
     * @return The data stream model data.
     */
    public static DataStream<IDFModelData> getModelDataStream(Table modelDataTable) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();
        return tEnv.toDataStream(modelDataTable)
                .map(x -> new IDFModelData(x.getFieldAs(0), x.getFieldAs(1), x.getFieldAs(2)));
    }

    /** Encoder for {@link IDFModelData}. */
    public static class ModelDataEncoder implements Encoder<IDFModelData> {
        private final DenseIntDoubleVectorSerializer denseVectorSerializer =
                new DenseIntDoubleVectorSerializer();

        @Override
        public void encode(IDFModelData modelData, OutputStream outputStream) throws IOException {
            DataOutputView dataOutputView = new DataOutputViewStreamWrapper(outputStream);
            denseVectorSerializer.serialize(modelData.idf, dataOutputView);
            LongPrimitiveArraySerializer.INSTANCE.serialize(modelData.docFreq, dataOutputView);
            LongSerializer.INSTANCE.serialize(modelData.numDocs, dataOutputView);
        }
    }

    /** Decoder for {@link IDFModelData}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<IDFModelData> {
        @Override
        public Reader<IDFModelData> createReader(Configuration config, FSDataInputStream stream) {
            return new Reader<IDFModelData>() {
                private final DenseIntDoubleVectorSerializer denseVectorSerializer =
                        new DenseIntDoubleVectorSerializer();

                @Override
                public IDFModelData read() throws IOException {
                    DataInputView source = new DataInputViewStreamWrapper(stream);
                    try {
                        DenseIntDoubleVector idf = denseVectorSerializer.deserialize(source);
                        long[] docFreq = LongPrimitiveArraySerializer.INSTANCE.deserialize(source);
                        long numDocs = LongSerializer.INSTANCE.deserialize(source);
                        return new IDFModelData(idf, docFreq, numDocs);
                    } catch (EOFException e) {
                        return null;
                    }
                }

                @Override
                public void close() throws IOException {
                    stream.close();
                }
            };
        }

        @Override
        public TypeInformation<IDFModelData> getProducedType() {
            return TypeInformation.of(IDFModelData.class);
        }
    }
}
