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

package org.apache.flink.ml.classification;

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.metrics.Gauge;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModelData;
import org.apache.flink.ml.classification.logisticregression.OnlineLogisticRegression;
import org.apache.flink.ml.classification.logisticregression.OnlineLogisticRegressionModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.InMemorySinkFunction;
import org.apache.flink.ml.util.InMemorySourceFunction;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.runtime.minicluster.MiniClusterConfiguration;
import org.apache.flink.runtime.testutils.InMemoryReporter;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.TestLogger;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.List;

import static org.apache.flink.ml.classification.logisticregression.OnlineLogisticRegressionModel.MODEL_DATA_VERSION_GAUGE_KEY;

/** Tests {@link OnlineLogisticRegression} and {@link OnlineLogisticRegressionModel}. */
public class FeedbackErrorTest extends TestLogger {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private static final double[] ONE_ARRAY = new double[] {1.0, 1.0, 1.0};

    private static final Row[] TRAIN_DENSE_ROWS_1 =
            new Row[] {
                Row.of(Vectors.dense(0.1, 2.), 0.),
                Row.of(Vectors.dense(0.2, 2.), 0.),
                Row.of(Vectors.dense(0.3, 2.), 0.),
                Row.of(Vectors.dense(0.4, 2.), 0.),
                Row.of(Vectors.dense(0.5, 2.), 0.),
                Row.of(Vectors.dense(11., 12.), 1.),
                Row.of(Vectors.dense(12., 11.), 1.),
                Row.of(Vectors.dense(13., 12.), 1.),
                Row.of(Vectors.dense(14., 12.), 1.),
                Row.of(Vectors.dense(15., 12.), 1.)
            };

    private static final Row[] PREDICT_DENSE_ROWS =
            new Row[] {
                Row.of(Vectors.dense(0.8, 2.7), 0.0), Row.of(Vectors.dense(15.5, 11.2), 1.0)
            };

    private static final Row[] PREDICT_SPARSE_ROWS =
            new Row[] {
                Row.of(Vectors.sparse(10, new int[] {1, 3, 5}, ONE_ARRAY), 0.),
                Row.of(Vectors.sparse(10, new int[] {5, 8, 9}, ONE_ARRAY), 1.)
            };

    private static final int defaultParallelism = 4;
    private static final int numTaskManagers = 2;
    private static final int numSlotsPerTaskManager = 2;

    private long currentModelDataVersion;

    private InMemorySourceFunction<Row> trainDenseSource;
    private InMemorySourceFunction<Row> predictDenseSource;
    private InMemorySinkFunction<Row> outputSink;
    private InMemorySinkFunction<LogisticRegressionModelData> modelDataSink;

    // TODO: creates static mini cluster once for whole test class after dependency upgrades to
    // Flink 1.15.
    private InMemoryReporter reporter;
    private MiniCluster miniCluster;
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;

    private Table onlineTrainDenseTable;
    private Table onlinePredictDenseTable;

    @Before
    public void before() throws Exception {
        currentModelDataVersion = 0;

        trainDenseSource = new InMemorySourceFunction<>();
        predictDenseSource = new InMemorySourceFunction<>();
        outputSink = new InMemorySinkFunction<>();
        modelDataSink = new InMemorySinkFunction<>();

        Configuration config = new Configuration();
        config.set(RestOptions.BIND_PORT, "18081-19091");
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        reporter = InMemoryReporter.create();
        reporter.addToConfiguration(config);

        miniCluster =
                new MiniCluster(
                        new MiniClusterConfiguration.Builder()
                                .setConfiguration(config)
                                .setNumTaskManagers(numTaskManagers)
                                .setNumSlotsPerTaskManager(numSlotsPerTaskManager)
                                .build());
        miniCluster.start();

        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.setParallelism(defaultParallelism);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);

        onlineTrainDenseTable =
                tEnv.fromDataStream(
                        env.addSource(
                                trainDenseSource,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            TypeInformation.of(DenseVector.class), Types.DOUBLE
                                        },
                                        new String[] {"features", "label"})));

        onlinePredictDenseTable =
                tEnv.fromDataStream(
                        env.addSource(
                                predictDenseSource,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            TypeInformation.of(DenseVector.class), Types.DOUBLE
                                        },
                                        new String[] {"features", "label"})));
    }

    @After
    public void after() throws Exception {
        miniCluster.close();
    }

    /**
     * Performs transform() on the provided model with predictTable, and adds sinks for
     * OnlineLogisticRegressionModel's transform output and model data.
     */
    private void transformAndOutputData(OnlineLogisticRegressionModel onlineModel) {
        Table outputTable = onlineModel.transform(onlinePredictDenseTable)[0];
        tEnv.toDataStream(outputTable).addSink(outputSink);

        Table modelDataTable = onlineModel.getModelData()[0];
        LogisticRegressionModelData.getModelDataStream(modelDataTable).addSink(modelDataSink);
    }

    /** Blocks the thread until Model has set up init model data. */
    private void waitInitModelDataSetup() throws InterruptedException {
        while (reporter.findMetrics(MODEL_DATA_VERSION_GAUGE_KEY).size() < defaultParallelism) {
            Thread.sleep(100);
        }
        waitModelDataUpdate();
    }

    /** Blocks the thread until the Model has received the next model-data-update event. */
    @SuppressWarnings("unchecked")
    private void waitModelDataUpdate() throws InterruptedException {
        do {
            long tmpModelDataVersion =
                    reporter.findMetrics(MODEL_DATA_VERSION_GAUGE_KEY).values().stream()
                            .map(x -> Long.parseLong(((Gauge<String>) x).getValue()))
                            .min(Long::compareTo)
                            .get();
            if (tmpModelDataVersion == currentModelDataVersion) {
                Thread.sleep(100);
            } else {
                currentModelDataVersion = tmpModelDataVersion;
                break;
            }
        } while (true);
    }

    /**
     * Inserts default predict data to the predict queue, fetches the prediction results, and
     * asserts that the grouping result is as expected.
     *
     * @param expectedRawInfo A list containing sets of expected result RawInfo.
     */
    private void predictAndAssert(List<DenseVector> expectedRawInfo) throws Exception {

        predictDenseSource.addAll(PREDICT_DENSE_ROWS);

        List<Row> rawResult = outputSink.poll(PREDICT_DENSE_ROWS.length);
        List<DenseVector> resultDetail = new ArrayList<>(rawResult.size());
        for (Row row : rawResult) {
            resultDetail.add(row.getFieldAs(3));
        }
        resultDetail.sort(TestUtils::compare);
        expectedRawInfo.sort(TestUtils::compare);
        for (int i = 0; i < resultDetail.size(); ++i) {
            double[] realData = resultDetail.get(i).values;
            double[] expectedData = expectedRawInfo.get(i).values;
            for (int j = 0; j < expectedData.length; ++j) {
                Assert.assertEquals(realData[j], expectedData[j], 1.0e-5);
            }
        }
    }

    @Test
    public void testFeedbackNum() throws Exception {
        OnlineLogisticRegression onlineLogisticRegression =
                new OnlineLogisticRegression()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setReg(0.2)
                        .setElasticNet(0.5)
                        .setGlobalBatchSize(10)
                        .setInitialModelData(
                                LogisticRegressionModelData.generateRandomModelData(tEnv, 2, 1));
        OnlineLogisticRegressionModel onlineModel =
                onlineLogisticRegression.fit(onlineTrainDenseTable);
        transformAndOutputData(onlineModel);
        Table modelTable = onlineModel.getModelData()[0];
        DataStream<Row> models = tEnv.toDataStream(modelTable);
        models.print();
        miniCluster.submitJob(env.getStreamGraph().getJobGraph());
        trainDenseSource.addAll(TRAIN_DENSE_ROWS_1);
        waitInitModelDataSetup();
        for (int i = 0; i < 10; ++i) {
            trainDenseSource.addAll(TRAIN_DENSE_ROWS_1);
            waitModelDataUpdate();
        }
    }

    @Test
    public void testFeedbackNumBug() throws Exception {
        OnlineLogisticRegression onlineLogisticRegression =
                new OnlineLogisticRegression()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setReg(0.2)
                        .setElasticNet(0.5)
                        .setGlobalBatchSize(10)
                        .setInitialModelData(
                                LogisticRegressionModelData.generateRandomModelData(tEnv, 2, 1));
        OnlineLogisticRegressionModel onlineModel =
                onlineLogisticRegression.fit(onlineTrainDenseTable);
        transformAndOutputData(onlineModel);
        Table modelTable = onlineModel.getModelData()[0];
        DataStream<Row> models = tEnv.toDataStream(modelTable);
        models.print();
        miniCluster.submitJob(env.getStreamGraph().getJobGraph());
        trainDenseSource.addAll(TRAIN_DENSE_ROWS_1);
        waitInitModelDataSetup();
        for (int i = 0; i < 10; ++i) {
            trainDenseSource.addAll(TRAIN_DENSE_ROWS_1);
        }
    }
}
