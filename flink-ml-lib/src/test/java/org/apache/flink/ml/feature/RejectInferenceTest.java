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

package org.apache.flink.ml.feature;

import org.apache.flink.ml.feature.rejectinference.RejectInference;
import org.apache.flink.ml.feature.rejectinference.RejectInferenceParams.RejectInferenceMethod;
import org.apache.flink.ml.feature.rejectinference.RejectInferenceParams.ScoreRangeMethod;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/** Test for {@link RejectInference}. */
public class RejectInferenceTest {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table accepts;
    private Table rejects;

    public String acceptRateCol = "acceptRateScoreCol";
    public String knownGoodBadCol = "knownGoodBadScoreCol";
    public String actualLabelCol = "label";

    private static final List<Row> ACCEPTS =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(0.2, 1.2, 0, 0, 1.0),
                            Row.of(0.4, 1.5, 1, 0, 1.0),
                            Row.of(0.6, 1.6, 2, 0, 1.0),
                            Row.of(0.8, 2.0, 3, 1, 1.0),
                            Row.of(1.0, 2.2, 4, 1, 1.0),
                            Row.of(1.2, 2.6, 5, 1, 1.0),
                            Row.of(1.4, 3.0, 6, 1, 1.0),
                            Row.of(1.6, 3.4, 7, 1, 0.5),
                            Row.of(1.8, 3.6, 8, 1, 0.5)));

    private static final List<Row> REJECTS =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(0.21, 1.0, 0),
                            Row.of(0.41, 1.2, 1),
                            Row.of(0.55, 1.7, 2),
                            Row.of(0.76, 1.9, 3),
                            Row.of(1.0, 2.0, 4),
                            Row.of(0.28, 1.0, 5),
                            Row.of(1.47, 2.2, 6),
                            Row.of(1.6, 2.4, 7),
                            Row.of(1.66, 2.5, 8)));

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        accepts =
                tEnv.fromDataStream(env.fromCollection(ACCEPTS))
                        .as(acceptRateCol, knownGoodBadCol, "userId", actualLabelCol, "weight");
        rejects =
                tEnv.fromDataStream(env.fromCollection(REJECTS))
                        .as(acceptRateCol, knownGoodBadCol, "userId");
    }

    private void compareResultAndExpected(List<Row> expectResult, List<Row> results) {
        results.sort(
                new Comparator<Row>() {
                    @Override
                    public int compare(Row o1, Row o2) {
                        int u1 = o1.getFieldAs("userId");
                        int u2 = o2.getFieldAs("userId");
                        if (u1 != u2) {
                            return u1 - u2;
                        }
                        int label1 = o1.getFieldAs("inferential_label");
                        int label2 = o2.getFieldAs("inferential_label");
                        return label1 - label2;
                    }
                });
        for (int i = 0; i < results.size(); i++) {
            Row result = results.get(i);
            Row expect = expectResult.get(i);
            assertEquals(expect.getField(0), result.getField("inferential_label"));
            assertEquals(
                    (double) expect.getField(1),
                    (double) result.getField("frequency_weight"),
                    1e-9);
            if (expect.getArity() == 3) {
                assertEquals(
                        (double) expect.getField(2),
                        (double) result.getField("knownGoodBadScoreCol"),
                        1e-9);
            }
        }
    }

    @Test
    public void testParam() {
        RejectInference rejectInference = new RejectInference();
        assertEquals("acceptRateScoreCol", rejectInference.getAcceptRateScoreCol());
        assertEquals("knownGoodBadScoreCol", rejectInference.getKnownGoodBadScoreCol());
        assertEquals("label", rejectInference.getLabelCol());
        assertNull(rejectInference.getWeightCol());
        assertEquals(RejectInferenceMethod.FUZZY, rejectInference.getRejectionInferenceMethod());
        assertEquals(ScoreRangeMethod.ACCEPTS, rejectInference.getScoreRangeMethod());
        assertEquals(0.3, rejectInference.getRejectionRate(), 1e-9);
        assertEquals(1.0, rejectInference.getEventRateIncrease(), 1e-9);
        assertNull(rejectInference.getCutoffScore());
        assertEquals(25, rejectInference.getNumBuckets());
        assertNull(rejectInference.getPdo());
        assertNull(rejectInference.getOdds());
        assertNull(rejectInference.getScaledValue());
        assertEquals(RejectInference.class.getName().hashCode(), rejectInference.getSeed());
        assertEquals(false, rejectInference.getWithScaled());

        rejectInference =
                rejectInference
                        .setAcceptRateScoreCol("col1")
                        .setKnownGoodBadScoreCol("col2")
                        .setLabelCol("col3")
                        .setWeightCol("col4")
                        .setRejectionInferenceMethod("PARCELLING")
                        .setScoreRangeMethod("REJECTS")
                        .setRejectionRate(0.1)
                        .setEventRateIncrease(1.1)
                        .setCutoffScore(500)
                        .setNumBuckets(20)
                        .setPdo(25)
                        .setOdds(50)
                        .setScaledValue(620)
                        .setSeed(1)
                        .setWithScaled(true);

        assertEquals("col1", rejectInference.getAcceptRateScoreCol());
        assertEquals("col2", rejectInference.getKnownGoodBadScoreCol());
        assertEquals("col3", rejectInference.getLabelCol());
        assertEquals("col4", rejectInference.getWeightCol());
        assertEquals(
                RejectInferenceMethod.PARCELLING, rejectInference.getRejectionInferenceMethod());
        assertEquals(ScoreRangeMethod.REJECTS, rejectInference.getScoreRangeMethod());
        assertEquals(0.1, rejectInference.getRejectionRate(), 1e-9);
        assertEquals(1.1, rejectInference.getEventRateIncrease(), 1e-9);
        assertEquals(500, rejectInference.getCutoffScore(), 1e-9);
        assertEquals(20, rejectInference.getNumBuckets());
        assertEquals(25, rejectInference.getPdo(), 1e-9);
        assertEquals(50, rejectInference.getOdds(), 1e-9);
        assertEquals(620, rejectInference.getScaledValue(), 1e-9);
        assertEquals(1, rejectInference.getSeed());
        assertEquals(true, rejectInference.getWithScaled());
    }

    @Test
    public void testFuzzy() {
        RejectInference rejectInference = new RejectInference();
        Table inferentialSamples = rejectInference.transform(accepts, rejects)[0];
        List<Row> expectedScoreRows =
                new ArrayList<>(
                        Arrays.asList(
                                Row.of(0, 0.11526060915856934),
                                Row.of(1, 0.31331081941285926),
                                Row.of(0, 0.09920366421470678),
                                Row.of(1, 0.3293677643567218),
                                Row.of(0, 0.06619939932151492),
                                Row.of(1, 0.3623720292499137),
                                Row.of(0, 0.055760774726999085),
                                Row.of(1, 0.3728106538444295),
                                Row.of(0, 0.051086966580907556),
                                Row.of(1, 0.37748446199052105),
                                Row.of(0, 0.11526060915856934),
                                Row.of(1, 0.31331081941285926),
                                Row.of(0, 0.042750209622722246),
                                Row.of(1, 0.38582121894870636),
                                Row.of(0, 0.03564544135453812),
                                Row.of(1, 0.3929259872168905),
                                Row.of(0, 0.03251064858053293),
                                Row.of(1, 0.3960607799908957)));

        List<Row> results = IteratorUtils.toList(inferentialSamples.execute().collect());
        compareResultAndExpected(expectedScoreRows, results);
    }

    @Test
    public void testTwoStage() {
        RejectInference rejectInference =
                new RejectInference()
                        .setWeightCol("weight")
                        .setRejectionInferenceMethod(RejectInferenceMethod.TWO_STAGE);
        Table inferentialSamples = rejectInference.transform(accepts, rejects)[0];
        double expectedFreqWeight = 0.380952380952381;
        List<Row> expectedScoreRows =
                new ArrayList<>(
                        Arrays.asList(
                                Row.of(0, expectedFreqWeight, 1.415005209048007),
                                Row.of(0, expectedFreqWeight, 1.5144611910094832),
                                Row.of(1, expectedFreqWeight, 1.8550323612591095),
                                Row.of(1, expectedFreqWeight, 1.9861142484614933),
                                Row.of(1, expectedFreqWeight, 2.05314338270992),
                                Row.of(0, expectedFreqWeight, 1.3986844901397406),
                                Row.of(1, expectedFreqWeight, 2.2072784761163007),
                                Row.of(1, expectedFreqWeight, 2.37278013191675),
                                Row.of(1, expectedFreqWeight, 2.4562542701768124)));
        List<Row> results = IteratorUtils.toList(inferentialSamples.execute().collect());
        compareResultAndExpected(expectedScoreRows, results);
    }

    @Test
    public void testAcceptsMethodParcelling() {
        env.setParallelism(1);
        int[] expectedLabels = {0, 0, 1, 0, 1, 1, 1};
        double expectedFreqWeight = 0.4285714285714286;
        List<Row> expectedScoreRows = new ArrayList<>(7);
        for (int i = 0; i < 7; i++) {
            expectedScoreRows.add(Row.of(expectedLabels[i], expectedFreqWeight));
        }
        RejectInference rejectInference =
                new RejectInference()
                        .setRejectionInferenceMethod(RejectInferenceMethod.PARCELLING)
                        .setScoreRangeMethod(ScoreRangeMethod.ACCEPTS)
                        .setNumBuckets(3);
        Table inferentialSamples = rejectInference.transform(accepts, rejects)[0];

        List<Row> results = IteratorUtils.toList(inferentialSamples.execute().collect());
        compareResultAndExpected(expectedScoreRows, results);
    }

    @Test
    public void testAugmentationMethodParcelling() {
        env.setParallelism(1);
        int[] expectedLabels = {0, 0, 0, 1, 1, 0, 1, 1, 1};
        double expectedFreqWeight = 1.0;
        List<Row> expectedScoreRows = new ArrayList<>(9);
        for (int i = 0; i < 9; i++) {
            expectedScoreRows.add(Row.of(expectedLabels[i], expectedFreqWeight));
        }
        RejectInference rejectInference =
                new RejectInference()
                        .setRejectionInferenceMethod(RejectInferenceMethod.PARCELLING)
                        .setScoreRangeMethod(ScoreRangeMethod.AUGMENTATION)
                        .setRejectionRate(0.5)
                        .setSeed(0)
                        .setNumBuckets(3);
        Table inferentialSamples = rejectInference.transform(accepts, rejects)[0];
        List<Row> results = IteratorUtils.toList(inferentialSamples.execute().collect());
        compareResultAndExpected(expectedScoreRows, results);
    }

    @Test
    public void testRejectsMethodParcelling() {
        env.setParallelism(1);
        int[] expectedLabels = {0, 0, 0, 0, 1, 0, 1, 1, 1};
        double expectedFreqWeight = 1.5;
        List<Row> expectedScoreRows = new ArrayList<>(9);
        for (int i = 0; i < 9; i++) {
            expectedScoreRows.add(Row.of(expectedLabels[i], expectedFreqWeight));
        }
        RejectInference rejectInference =
                new RejectInference()
                        .setRejectionInferenceMethod(RejectInferenceMethod.PARCELLING)
                        .setScoreRangeMethod(ScoreRangeMethod.REJECTS)
                        .setRejectionRate(0.6)
                        .setNumBuckets(3);
        Table inferentialSamples = rejectInference.transform(accepts, rejects)[0];
        List<Row> results = IteratorUtils.toList(inferentialSamples.execute().collect());
        compareResultAndExpected(expectedScoreRows, results);
    }

    @Test
    public void testHardCutoff() {
        int[] expectedLabels = {0, 0, 0, 0, 1, 0, 1, 1, 1};
        double expectedFreqWeight = 0.4285714285714286;
        List<Row> expectedScoreRows = new ArrayList<>(9);
        for (int i = 0; i < 9; i++) {
            expectedScoreRows.add(Row.of(expectedLabels[i], expectedFreqWeight));
        }
        RejectInference rejectInference =
                new RejectInference()
                        .setRejectionInferenceMethod(RejectInferenceMethod.HARD_CUTOFF)
                        .setCutoffScore(2);
        Table inferentialSamples = rejectInference.transform(accepts, rejects)[0];
        List<Row> results = IteratorUtils.toList(inferentialSamples.execute().collect());
        compareResultAndExpected(expectedScoreRows, results);
    }

    @Test
    public void testWeightCol() {
        int[] expectedLabels = {0, 0, 0, 0, 1, 0, 1, 1, 1};
        double expectedFreqWeight = 0.380952380952381;
        List<Row> expectedScoreRows = new ArrayList<>(9);
        for (int i = 0; i < 9; i++) {
            expectedScoreRows.add(Row.of(expectedLabels[i], expectedFreqWeight));
        }
        RejectInference rejectInference =
                new RejectInference()
                        .setRejectionInferenceMethod(RejectInferenceMethod.HARD_CUTOFF)
                        .setWeightCol("weight")
                        .setCutoffScore(2);
        Table inferentialSamples = rejectInference.transform(accepts, rejects)[0];
        List<Row> results = IteratorUtils.toList(inferentialSamples.execute().collect());
        compareResultAndExpected(expectedScoreRows, results);
    }
}
