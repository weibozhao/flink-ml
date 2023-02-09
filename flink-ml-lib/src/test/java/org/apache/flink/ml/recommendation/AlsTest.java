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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/** Tests {@link Als} and {@link AlsModel}. */
public class AlsTest extends AbstractTestBase {

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;

    private StreamTableEnvironment tEnv;

    private static final List<Row> trainData =
            Arrays.asList(
                    Row.of(1L, 5L, 0.7),
                    Row.of(2L, 7L, 0.4),
                    Row.of(1L, 8L, 0.3),
                    Row.of(4L, 6L, 0.4),
                    Row.of(3L, 7L, 0.6),
                    Row.of(1L, 6L, 0.5),
                    Row.of(4L, 8L, 0.3),
                    Row.of(2L, 6L, 0.4));

    private static final double[] expectedCoefficient = new double[] {1.141, 1.829};

    private static final double TOLERANCE = 1e-7;

    private static final double PREDICTION_TOLERANCE = 0.1;

    private static final double COEFFICIENT_TOLERANCE = 0.1;

    private Table trainDataTable;

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.getConfig().enableObjectReuse();
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);
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
    }

    @SuppressWarnings("unchecked")
    private void verifyPredictionResult(Table output, String labelCol, String predictionCol)
            throws Exception {
        List<Row> predResult = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        for (Row predictionRow : predResult) {
            double label = ((Number) predictionRow.getField(labelCol)).doubleValue();
            double prediction = (double) predictionRow.getField(predictionCol);
            assertTrue(Math.abs(prediction - label) / label < PREDICTION_TOLERANCE);
        }
    }

    @Test
    public void testParam() {
        Als als = new Als();
        assertEquals("user", als.getUserCol());
        assertEquals("item", als.getItemCol());
        assertNull(als.getRatingCol());
        assertEquals(1.0, als.getAlpha(), TOLERANCE);
        assertEquals(0.1, als.getRegParam(), TOLERANCE);
        assertEquals(10, als.getRank());
        assertEquals(false, als.getImplicitprefs());
        assertEquals(false, als.getNonnegative());
        assertEquals(10, als.getMaxIter());
        assertEquals(10, als.getNumUserBlocks());
        assertEquals(10, als.getNumItemBlocks());
        assertEquals("predictionCol", als.getPredictionCol());

        als.setUserCol("userCol")
                .setItemCol("itemCol")
                .setRatingCol("rating")
                .setAlpha(0.001)
                .setRegParam(0.5)
                .setRank(100)
                .setImplicitprefs(true)
                .setNonnegative(true)
                .setMaxIter(1000)
                .setNumUserBlocks(5)
                .setNumItemBlocks(5)
                .setPredictionCol("pred");

        assertEquals("userCol", als.getUserCol());
        assertEquals("itemCol", als.getItemCol());
        assertEquals("rating", als.getRatingCol());
        assertEquals(0.001, als.getAlpha(), TOLERANCE);
        assertEquals(0.5, als.getRegParam(), TOLERANCE);
        assertEquals(100, als.getRank());
        assertEquals(true, als.getImplicitprefs());
        assertEquals(true, als.getNonnegative());
        assertEquals(1000, als.getMaxIter());
        assertEquals(5, als.getNumUserBlocks());
        assertEquals(5, als.getNumItemBlocks());
        assertEquals("pred", als.getPredictionCol());
    }

    // @Test
    // public void testOutputSchema() {
    //	Table tempTable = trainDataTable.as("test_features", "test_label", "test_weight");
    //	Als als =
    //		new Als()
    //			.setFeaturesCol("test_features")
    //			.setLabelCol("test_label")
    //			.setWeightCol("test_weight")
    //			.setPredictionCol("test_predictionCol");
    //	Table output = als.fit(trainDataTable).transform(tempTable)[0];
    //	assertEquals(
    //		Arrays.asList("test_features", "test_label", "test_weight", "test_predictionCol"),
    //		output.getResolvedSchema().getColumnNames());
    // }
    //
    // @Test
    // public void testFitAndPredict() throws Exception {
    //	Als als = new Als().setWeightCol("weight");
    //	Table output = als.fit(trainDataTable).transform(trainDataTable)[0];
    //	verifyPredictionResult(
    //		output, als.getLabelCol(), als.getPredictionCol());
    // }
    //
    // @Test
    // public void testInputTypeConversion() throws Exception {
    //	trainDataTable = TestUtils.convertDataTypesToSparseInt(tEnv, trainDataTable);
    //	assertArrayEquals(
    //		new Class <?>[] {SparseVector.class, Integer.class, Integer.class},
    //		TestUtils.getColumnDataTypes(trainDataTable));
    //
    //	Als als = new Als().setWeightCol("weight");
    //	Table output = als.fit(trainDataTable).transform(trainDataTable)[0];
    //	verifyPredictionResult(
    //		output, als.getLabelCol(), als.getPredictionCol());
    // }
    //
    // @Test
    // public void testSaveLoadAndPredict() throws Exception {
    //	Als als = new Als().setWeightCol("weight");
    //	als =
    //		TestUtils.saveAndReload(
    //			tEnv, als, tempFolder.newFolder().getAbsolutePath());
    //	AlsModel model = als.fit(trainDataTable);
    //	model = TestUtils.saveAndReload(tEnv, model, tempFolder.newFolder().getAbsolutePath());
    //	assertEquals(
    //		Collections.singletonList("coefficient"),
    //		model.getModelData()[0].getResolvedSchema().getColumnNames());
    //	Table output = model.transform(trainDataTable)[0];
    //	verifyPredictionResult(
    //		output, als.getLabelCol(), als.getPredictionCol());
    // }
    //
    // @Test
    // public void testGetModelData() throws Exception {
    //	Als als = new Als().setWeightCol("weight");
    //	AlsModel model = als.fit(trainDataTable);
    //	List <AlsModelData> modelData =
    //		IteratorUtils.toList(
    //			AlsModelData.getModelDataStream(model.getModelData()[0])
    //				.executeAndCollect());
    //	assertNotNull(modelData);
    //	assertEquals(1, modelData.size());
    //	assertArrayEquals(
    //		expectedCoefficient, modelData.get(0).coefficient.values, COEFFICIENT_TOLERANCE);
    // }
    //
    // @Test
    // public void testSetModelData() throws Exception {
    //	Als als = new Als().setWeightCol("weight");
    //	AlsModel model = als.fit(trainDataTable);
    //
    //	AlsModel newModel = new AlsModel();
    //	ReadWriteUtils.updateExistingParams(newModel, model.getParamMap());
    //	newModel.setModelData(model.getModelData());
    //	Table output = newModel.transform(trainDataTable)[0];
    //	verifyPredictionResult(
    //		output, als.getLabelCol(), als.getPredictionCol());
    // }
}
