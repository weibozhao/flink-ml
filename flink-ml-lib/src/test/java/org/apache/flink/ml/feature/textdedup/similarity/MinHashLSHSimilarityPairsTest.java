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

import org.apache.flink.api.common.ExecutionMode;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static org.apache.flink.table.api.Expressions.$;

/** Tests for MinHashLSHSimilarityPairs. */
public class MinHashLSHSimilarityPairsTest extends AbstractTestBase {

    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;

    private Table inputTable;

    @Before
    public void before() {
        Configuration configuration = new Configuration();
        configuration.setString("execution.runtime-mode", "BATCH");
        env = TestUtils.getExecutionEnvironment(configuration);
        //        env.setParallelism(4);
        env.getConfig().setExecutionMode(ExecutionMode.BATCH);
        //        env.getConfig().disableGenericTypes();
        //        env.getConfig().enableObjectReuse();
        tEnv = StreamTableEnvironment.create(env);

        List<Row> inputRows =
                Arrays.asList(
                        Row.of(
                                0L,
                                Vectors.sparse(6, new int[] {0, 1, 2}, new double[] {1., 1., 1.})),
                        Row.of(
                                1L,
                                Vectors.sparse(6, new int[] {2, 3, 4}, new double[] {1., 1., 1.})),
                        Row.of(
                                2L,
                                Vectors.sparse(6, new int[] {0, 2, 4}, new double[] {1., 1., 1.})));

        Schema schema =
                Schema.newBuilder()
                        .column("f0", DataTypes.BIGINT())
                        .column("f1", DataTypes.of(SparseVector.class))
                        .build();
        DataStream<Row> dataStream = env.fromCollection(inputRows);

        inputTable = tEnv.fromDataStream(dataStream, schema).as("id", "vec");
    }

    @Test
    public void testMinHashFunction() {
        int numHashTables = 2;
        int numProjectionsPerTable = 3;
        MinHashFunction minHashFunction =
                new MinHashFunction(0, numProjectionsPerTable, numHashTables);
        Vector vec = Vectors.sparse(10, new int[] {2, 3, 5, 7}, new double[] {1., 1., 1., 1.});
        Assert.assertArrayEquals(
                new long[] {-866562137152317294L, 388866554276253505L},
                minHashFunction.hashFunctionToLong(vec));
    }

    @Test
    public void testMinHash() throws Exception {
        MinHash minHash =
                new MinHash()
                        .setInputCol("vec")
                        .setSeed(2022L)
                        .setNumHashTables(2)
                        .setNumHashFunctionsPerTable(3);
        Table output = minHash.transform(inputTable)[0].select($("id"), $(minHash.getOutputCol()));
        List<Row> expected =
                Arrays.asList(
                        Row.of(0L, new long[] {-1514938083922410922L, -3928666922037747718L}),
                        Row.of(1L, new long[] {-8719543354350720237L, 4816915348192523648L}),
                        Row.of(2L, new long[] {-8719543354350720237L, -9193398650638803871L}));
        //noinspection unchecked
        List<Row> results = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        compareResultCollections(expected, results, Comparator.comparingLong(d -> d.getFieldAs(0)));
    }

    @Test
    public void testMinHashLSHSimilarityPairs() {
        String idCol = "id";
        String vecCol = "vec";

        String minHashOutputCol = "minHashOutput";
        MinHash vectorMinHash =
                new MinHash()
                        .setInputCol(vecCol)
                        .setOutputCol(minHashOutputCol)
                        .setSeed(2022L)
                        .setNumHashTables(2)
                        .setNumHashFunctionsPerTable(3);
        Table minHashOutput =
                vectorMinHash.transform(inputTable)[0].select(
                        $(idCol), $(vecCol), $(minHashOutputCol));

        MinHashLSHSimilarityPairs similarityJoin =
                new MinHashLSHSimilarityPairs()
                        .setIdCol(idCol)
                        .setVectorCol(vecCol)
                        .setMinHashSignatureCol(minHashOutputCol)
                        .setThreshold(0.6)
                        .setBucketChunkSize(4096)
                        .setItemCandidateLimitInBucket(4096);
        Table result = similarityJoin.transform(new Table[] {minHashOutput})[0];
        Table distinct = result.select($("id1"), $("id2")).distinct();
        @SuppressWarnings("unchecked")
        List<Row> list = IteratorUtils.toList(distinct.execute().collect());
        Assert.assertEquals(1, list.size());
    }
}
