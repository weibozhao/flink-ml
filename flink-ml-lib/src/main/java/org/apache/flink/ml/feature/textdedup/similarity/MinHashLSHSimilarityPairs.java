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

package org.apache.flink.ml.feature.textdedup.similarity;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.apache.flink.table.api.Expressions.$;

/** Extracts similar vectors using the MinHashLSH algorithm. */
public class MinHashLSHSimilarityPairs
        implements MinHashLSHSimilarityPairsParams<MinHashLSHSimilarityPairs>,
                AlgoOperator<MinHashLSHSimilarityPairs> {

    private static final Logger LOG = LoggerFactory.getLogger(MinHashLSHSimilarityPairs.class);

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public MinHashLSHSimilarityPairs() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    static int intersect(int[] indices1, int[] indices2) {
        int index1 = 0;
        int index2 = 0;
        int intersect = 0;
        while (index1 < indices1.length && index2 < indices2.length) {
            if (indices1[index1] == indices2[index2]) {
                intersect++;
                index1++;
                index2++;
            } else if (indices1[index1] < indices2[index2]) {
                index1++;
            } else {
                index2++;
            }
        }
        return intersect;
    }

    @Override
    public Table[] transform(Table[] inputs) {
        Table in = inputs[0];
        final StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        if (null != getIncrementalIndicatorCol()) {
            in =
                    in.select(
                            $(getIdCol()),
                            $(getMinHashSignatureCol()),
                            $(getVectorCol()),
                            $(getIncrementalIndicatorCol()));
        } else {
            in = in.select($(getIdCol()), $(getMinHashSignatureCol()), $(getVectorCol()));
        }

        DataStream<Row> data = tEnv.toDataStream(in);
        KeyedStream<Tuple4<Long, Long, int[], Boolean>, Long> buckets =
                data.map(new DecodeMinHashMapper())
                        .returns(
                                Types.TUPLE(
                                        Types.LONG,
                                        Types.PRIMITIVE_ARRAY(Types.LONG),
                                        Types.PRIMITIVE_ARRAY(Types.INT),
                                        Types.BOOLEAN))
                        .name("DecodeMinHash")
                        .flatMap(new BucketingMinHashFlatMapper())
                        .name("BucketingMinHash")
                        .keyBy(d -> d.f0);

        DataStream<Tuple3<Long, Long, Double>> similarPairs =
                buckets.transform(
                                "GenerateChunksOperator",
                                Types.TUPLE(
                                        Types.INT,
                                        Types.PRIMITIVE_ARRAY(Types.LONG),
                                        Types.OBJECT_ARRAY(Types.PRIMITIVE_ARRAY(Types.INT)),
                                        Types.PRIMITIVE_ARRAY(Types.BOOLEAN)),
                                new GenerateChunksOperator(getBucketChunkSize()))
                        .name("GenerateChunks")
                        .rebalance()
                        .flatMap(
                                new FindChunkSimilarityMapper(
                                        getThreshold(), getItemCandidateLimitInBucket()))
                        .name("FindChunkSimilarity");

        DataStream<Row> output =
                similarPairs.map(
                        d -> Row.of(d.f0, d.f1, d.f2),
                        Types.ROW_NAMED(
                                new String[] {"id1", "id2", "dist"},
                                Types.LONG,
                                Types.LONG,
                                Types.DOUBLE));
        return new Table[] {tEnv.fromDataStream(output)};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    private static class GenerateChunksOperator
            extends AbstractStreamOperator<Tuple4<Integer, long[], int[][], boolean[]>>
            implements OneInputStreamOperator<
                            Tuple4<Long, Long, int[], Boolean>,
                            Tuple4<Integer, long[], int[][], boolean[]>>,
                    BoundedOneInput {

        private final int chunkSize;
        private Long lastKey;
        private int counter = 0;
        private transient long[] ids;
        private transient int[][] indices;
        private transient boolean[] indicators;

        public GenerateChunksOperator(int bucketChunkSize) {
            chunkSize = bucketChunkSize;
        }

        @Override
        public void open() throws Exception {
            super.open();
            ids = new long[chunkSize];
            indices = new int[chunkSize][];
            indicators = new boolean[chunkSize];
            LOG.info("chunkSize: {}", chunkSize);
            // counter will never be larger than chunkSize.
            counter = 0;
        }

        @Override
        public void endInput() {
            if (null != lastKey) {
                checkCurrentKey();
            }
            counter = 0;
        }

        private void checkCurrentKey() {
            if ((counter > 1)) {
                for (int i = counter; i < chunkSize; i += 1) {
                    indices[i] = new int[0];
                }
                output.collect(new StreamRecord<>(Tuple4.of(counter, ids, indices, indicators)));
            }
            if (counter > 8192) {
                LOG.info("Large chunk with key {} encountered: {}", lastKey, counter);
            }
            counter = 0;
        }

        @Override
        public void processElement(StreamRecord<Tuple4<Long, Long, int[], Boolean>> streamRecord) {
            Tuple4<Long, Long, int[], Boolean> value = streamRecord.getValue();
            long key = value.f0;
            if (null == lastKey) {
                lastKey = key;
            }
            // Switch to new chunks when key changed or `chunkSize` reached.
            if (key != lastKey) {
                checkCurrentKey();
                lastKey = key;
                counter = 0;
            }
            ids[counter % chunkSize] = value.f1;
            indices[counter % chunkSize] = value.f2;
            indicators[counter % chunkSize] = value.f3;
            counter += 1;
            if (counter == chunkSize) {
                output.collect(new StreamRecord<>(Tuple4.of(counter, ids, indices, indicators)));
                counter = 0;
            }
        }
    }

    private static class FindChunkSimilarityMapper
            extends RichFlatMapFunction<
                    Tuple4<Integer, long[], int[][], boolean[]>, Tuple3<Long, Long, Double>> {

        private final double thresh;
        private final int itemCandidateLimitInBucket;

        public FindChunkSimilarityMapper(double thresh, Integer itemCandidateLimitInBucket) {
            this.thresh = thresh;
            this.itemCandidateLimitInBucket = itemCandidateLimitInBucket;
        }

        @Override
        public void open(Configuration parameters) {
            LOG.info("itemCandidateLimitInBucket: {}", itemCandidateLimitInBucket);
        }

        @Override
        public void flatMap(
                Tuple4<Integer, long[], int[][], boolean[]> value,
                Collector<Tuple3<Long, Long, Double>> out) {
            int chunkSize = value.f0;
            long[] ids = value.f1;
            int[][] indices = value.f2;
            boolean[] indicators = value.f3;

            final long windowSize = Math.min(itemCandidateLimitInBucket, chunkSize);

            for (int i = 0; i < chunkSize; i += 1) {
                for (int j = i + 1; j < i + windowSize; j += 1) {
                    int modJ = j % chunkSize;
                    // If both two items are old ones, skip this pair.
                    if (!indicators[i] && !indicators[modJ]) {
                        continue;
                    }
                    // Force ids[i] < ids[modJ] to avoid duplicated pairs.
                    if (ids[i] >= ids[modJ]) {
                        continue;
                    }
                    int intersect = intersect(indices[i], indices[modJ]);
                    double dist =
                            1.
                                    - 1.
                                            * intersect
                                            / (indices[i].length
                                                    - intersect
                                                    + indices[modJ].length);
                    if (dist < thresh) {
                        out.collect(Tuple3.of(ids[i], ids[modJ], dist));
                    }
                }
            }
        }
    }

    static class DecodeMinHashMapper
            implements MapFunction<Row, Tuple4<Long, long[], int[], Boolean>> {

        @Override
        public Tuple4<Long, long[], int[], Boolean> map(Row in) throws Exception {
            long[] arr = in.getFieldAs(1);
            SparseVector sp = ((Vector) in.getFieldAs(2)).toSparse();
            boolean isNew = in.getArity() == 3 || (Boolean) in.getFieldAs(3);
            return Tuple4.of((Long) in.getField(0), arr, sp.indices, isNew);
        }
    }

    static class BucketingMinHashFlatMapper
            implements FlatMapFunction<
                    Tuple4<Long, long[], int[], Boolean>, Tuple4<Long, Long, int[], Boolean>> {

        @Override
        public void flatMap(
                Tuple4<Long, long[], int[], Boolean> value,
                Collector<Tuple4<Long, Long, int[], Boolean>> out)
                throws Exception {
            Long id = value.f0;
            long[] minHash = value.f1;
            long minHashLen = minHash.length;
            for (int i = 0; i < minHash.length; i += 1) {
                out.collect(Tuple4.of(minHash[i] * minHashLen + i, id, value.f2, value.f3));
            }
        }
    }
}
