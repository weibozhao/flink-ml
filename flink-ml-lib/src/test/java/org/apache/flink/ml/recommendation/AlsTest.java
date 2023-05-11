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

package org.apache.flink.ml.recommendation;

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.recommendation.als.Als;
import org.apache.flink.ml.recommendation.als.AlsModel;
import org.apache.flink.ml.recommendation.als.AlsRating;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/** Tests {@link Als} and {@link AlsModel}. */
public class AlsTest extends AbstractTestBase {

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;

    private StreamTableEnvironment tEnv;

    private List<Row> trainData =
            Arrays.asList(
                    Row.of(1L, 5L, 0.7),
                    Row.of(2L, 7L, 0.4),
                    Row.of(1L, 8L, 0.3),
                    Row.of(4L, 6L, 0.4),
                    Row.of(3L, 7L, 0.6),
                    Row.of(1L, 6L, 0.5),
                    Row.of(4L, 8L, 0.3),
                    Row.of(2L, 6L, 0.4));

    private static final double TOLERANCE = 1e-7;

    private Table trainDataTable;
    private Table testDataTable;

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.getConfig().enableObjectReuse();
        env.setParallelism(2);
        // env.setRuntimeMode(RuntimeExecutionMode.BATCH);
        // env.setStateBackend(new EmbeddedRocksDBStateBackend());
        env.getCheckpointConfig().disableCheckpointing(); //
       // env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        // env.setRuntimeMode(RuntimeExecutionMode.BATCH);
        tEnv = StreamTableEnvironment.create(env);
        Random rand = new Random(0);
        List<Row> trainData = new ArrayList<>();
        for (int i = 0; i < 4000000; ++i) {
            trainData.add(
                    Row.of((long) rand.nextInt(500000), (long) rand.nextInt(500000), rand.nextDouble()));
        }

        Collections.shuffle(trainData);

        List<Row> testData = new ArrayList<>();
        testData.add(Row.of(2L, 4L));

        Collections.shuffle(trainData);
        trainDataTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                trainData,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            Types.LONG, Types.LONG, Types.DOUBLE
                                        },
                                        new String[] {"user_id", "item_id", "rating"})));

        testDataTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                testData,
                                new RowTypeInfo(
                                        new TypeInformation[] {Types.LONG, Types.LONG},
                                        new String[] {"user_id", "item_id"})));
    }

    @SuppressWarnings("unchecked")
    private void verifyPredictionResult(Table output, String ratingCol, String predictionCol)
            throws Exception {
        List<Row> predResult = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        for (Row predictionRow : predResult) {
            long label = predictionRow.getFieldAs(ratingCol);
            double prediction = (double) predictionRow.getField(predictionCol);
            System.out.println(label + " " + prediction);
            // assertTrue(Math.abs(prediction - label) / label < PREDICTION_TOLERANCE);
        }
    }

    @Test
    public void testParam() {
        Als als = new Als();
        assertEquals("user", als.getUserCol());
        assertEquals("item", als.getItemCol());
        assertEquals("rating", als.getRatingCol());
        assertEquals(1.0, als.getAlpha(), TOLERANCE);
        assertEquals(0.1, als.getRegParam(), TOLERANCE);
        assertEquals(10, als.getRank());
        assertEquals(false, als.getImplicitPrefs());
        assertEquals(false, als.getNonNegative());
        assertEquals(10, als.getMaxIter());
        assertEquals(10, als.getNumUserBlocks());
        assertEquals(10, als.getNumItemBlocks());
        assertEquals("prediction", als.getPredictionCol());

        als.setUserCol("userCol")
                .setItemCol("itemCol")
                .setRatingCol("ratingCol")
                .setAlpha(0.001)
                .setRegParam(0.5)
                .setRank(100)
                .setImplicitPrefs(false)
                .setNonNegative(true)
                .setMaxIter(1000)
                .setNumUserBlocks(5)
                .setNumItemBlocks(5)
                .setPredictionCol("pred");

        assertEquals("userCol", als.getUserCol());
        assertEquals("itemCol", als.getItemCol());
        assertEquals("ratingCol", als.getRatingCol());
        assertEquals(0.001, als.getAlpha(), TOLERANCE);
        assertEquals(0.5, als.getRegParam(), TOLERANCE);
        assertEquals(100, als.getRank());
        assertEquals(false, als.getImplicitPrefs());
        assertEquals(true, als.getNonNegative());
        assertEquals(1000, als.getMaxIter());
        assertEquals(5, als.getNumUserBlocks());
        assertEquals(5, als.getNumItemBlocks());
        assertEquals("pred", als.getPredictionCol());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable = trainDataTable.as("user_id", "item_id", "rating_col");
        Als als =
                new Als()
                        .setSeed(1)
                        .setUserCol("user_id")
                        .setItemCol("item_id")
                        .setRatingCol("rating_col")
                        .setPredictionCol("pred");
        AlsModel model = als.fit(trainDataTable);
        Table output = model.transform(tempTable)[0];
        assertEquals(
                Arrays.asList("user_id", "item_id", "rating_col", "pred"),
                output.getResolvedSchema().getColumnNames());

        AlsRating alsRating =
                new AlsRating()
                        .setSeed(1)
                        .setUserCol("user_id")
                        .setItemCol("item_id")
                        .setRatingCol("rating_col")
                        .setPredictionCol("pred");
        AlsModel ratingModel = alsRating.fit(trainDataTable);
        Table outputRating = ratingModel.transform(tempTable)[0];
        assertEquals(
                Arrays.asList("user_id", "item_id", "rating_col", "pred"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredict() throws Exception {
        for (int i = 1; i < 2; ++i) {
            //AlsRating alsRating =
            //        new AlsRating()
            //                .setUserCol("user_id")
            //                .setItemCol("item_id")
            //                .setRatingCol("rating")
            //                .setNumUserBlocks(i)
            //                .setMaxIter(2)
            //                .setRank(10)
            //                .setAlpha(0.1)
            //                .setRegParam(0.1)
            //                .setSeed(0)
            //                .setImplicitPrefs(true)
            //                .setNonNegative(true)
            //                .setNumItemBlocks(i)
            //                .setPredictionCol("pred");
            //Table output = alsRating.fit(trainDataTable).transform(testDataTable)[0];
            //verifyPredictionResult(output, alsRating.getItemCol(), alsRating.getPredictionCol());

            Als als =
                    new Als()
                            .setUserCol("user_id")
                            .setItemCol("item_id")
                            .setRatingCol("rating")
                            .setNumUserBlocks(i)
                            .setMaxIter(2)
                            .setRank(10)
                            .setAlpha(0.1)
                            .setRegParam(0.1)
                            .setSeed(0)
                            .setImplicitPrefs(true)
                            .setNonNegative(true)
                            .setNumItemBlocks(i)
                            .setPredictionCol("pred");
            Table output1 = als.fit(trainDataTable).transform(testDataTable)[0];
            verifyPredictionResult(output1, als.getItemCol(), als.getPredictionCol());
        }
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        AlsRating als =
                new AlsRating()
                        .setUserCol("user_id")
                        .setItemCol("item_id")
                        .setRatingCol("rating")
                        .setNumUserBlocks(1)
                        .setMaxIter(1)
                        .setRank(10)
                        .setAlpha(0.1)
                        .setRegParam(0.1)
                        .setSeed(0)
                        .setImplicitPrefs(true)
                        .setNonNegative(true)
                        .setNumItemBlocks(1)
                        .setPredictionCol("pred");
        AlsModel model = als.fit(trainDataTable);
        Table output = model.transform(testDataTable)[0];
        verifyPredictionResult(output, als.getItemCol(), als.getPredictionCol());
        // AlsModel loadModel =
        //        TestUtils.saveAndReload(tEnv, model, tempFolder.newFolder().getAbsolutePath());

        // Table output1 = loadModel.transform(testDataTable)[0];
        // verifyPredictionResult(output1, als.getItemCol(), als.getPredictionCol());
    }
}
