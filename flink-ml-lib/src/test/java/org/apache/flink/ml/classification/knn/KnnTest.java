package org.apache.flink.ml.classification.knn;

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.api.Stage;
import org.apache.flink.ml.builder.Pipeline;
import org.apache.flink.ml.common.param.HasFeaturesCol;
import org.apache.flink.ml.common.param.HasK;
import org.apache.flink.ml.common.param.HasLabelCol;
import org.apache.flink.ml.common.param.HasPredictionCol;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.param.Param;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/** knn algorithm test. */
public class KnnTest {
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table trainData;

    List<Row> trainArray =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of("f", Vectors.dense(2.0, 3.0)),
                            Row.of("f", Vectors.dense(2.1, 3.1)),
                            Row.of("m", Vectors.dense(200.1, 300.1)),
                            Row.of("m", Vectors.dense(200.2, 300.2)),
                            Row.of("m", Vectors.dense(200.3, 300.3)),
                            Row.of("m", Vectors.dense(200.4, 300.4)),
                            Row.of("m", Vectors.dense(200.4, 300.4)),
                            Row.of("m", Vectors.dense(200.6, 300.6)),
                            Row.of("f", Vectors.dense(2.1, 3.1)),
                            Row.of("f", Vectors.dense(2.1, 3.1)),
                            Row.of("f", Vectors.dense(2.1, 3.1)),
                            Row.of("f", Vectors.dense(2.1, 3.1)),
                            Row.of("f", Vectors.dense(2.3, 3.2)),
                            Row.of("f", Vectors.dense(2.3, 3.2)),
                            Row.of("c", Vectors.dense(2.8, 3.2)),
                            Row.of("d", Vectors.dense(300., 3.2)),
                            Row.of("f", Vectors.dense(2.2, 3.2)),
                            Row.of("e", Vectors.dense(2.4, 3.2)),
                            Row.of("e", Vectors.dense(2.5, 3.2)),
                            Row.of("e", Vectors.dense(2.5, 3.2)),
                            Row.of("f", Vectors.dense(2.1, 3.1))));

    List<Row> testArray =
            new ArrayList<>(
                    Arrays.asList(Row.of(Vectors.dense(4.0, 4.1)), Row.of(Vectors.dense(300, 42))));
    private Table testData;

    Row[] expectedData =
            new Row[] {Row.of("e", Vectors.dense(4.0, 4.1)), Row.of("m", Vectors.dense(300, 42))};

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);

        Schema schema =
                Schema.newBuilder()
                        .column("f0", DataTypes.STRING())
                        .column("f1", DataTypes.of(DenseVector.class))
                        .build();

        DataStream<Row> dataStream = env.fromCollection(trainArray);
        trainData = tEnv.fromDataStream(dataStream, schema).as("label, vec");

        Schema outputSchema =
                Schema.newBuilder().column("f0", DataTypes.of(DenseVector.class)).build();

        DataStream<Row> predDataStream = env.fromCollection(testArray);
        testData = tEnv.fromDataStream(predDataStream, outputSchema).as("vec");
    }

    /** test knn Estimator. */
    @Test
    public void testFitAntTransform() throws Exception {
        Knn knn =
                new Knn()
                        .setLabelCol("label")
                        .setFeaturesCol("vec")
                        .setK(4)
                        .setPredictionCol("pred");

        KnnModel knnModel = knn.fit(trainData);
        Table result = knnModel.transform(testData)[0];

        DataStream<Row> output = tEnv.toDataStream(result);

        List<Row> rows = IteratorUtils.toList(output.executeAndCollect());
        for (Row value : rows) {
            for (Row exp : expectedData) {
                assert !exp.getField(1).equals(value.getField(0))
                        || (exp.getField(0).equals(value.getField(1)));
            }
        }
    }

    /** test knn Estimator. */
    @Test
    public void testParamsConstructor() throws Exception {
        Map<Param<?>, Object> params = new HashMap<>();
        params.put(HasLabelCol.LABEL_COL, "label");
        params.put(HasFeaturesCol.FEATURES_COL, "vec");
        params.put(HasK.K, 4);
        params.put(HasPredictionCol.PREDICTION_COL, "pred");
        Knn knn = new Knn(params);

        KnnModel knnModel = knn.fit(trainData);
        Table result = knnModel.transform(testData)[0];

        DataStream<Row> output = tEnv.toDataStream(result);

        List<Row> rows = IteratorUtils.toList(output.executeAndCollect());
        for (Row value : rows) {
            for (Row exp : expectedData) {
                assert !exp.getField(1).equals(value.getField(0))
                        || (exp.getField(0).equals(value.getField(1)));
            }
        }
    }

    /** test knn as a pipeline stage. */
    @Test
    public void testPipeline() throws Exception {
        Knn knn =
                new Knn()
                        .setLabelCol("label")
                        .setFeaturesCol("vec")
                        .setK(4)
                        .setPredictionCol("pred");

        List<Stage<?>> stages = new ArrayList<>();
        stages.add(knn);

        Pipeline pipe = new Pipeline(stages);

        Table result = pipe.fit(trainData).transform(testData)[0];

        DataStream<Row> output = tEnv.toDataStream(result);

        List<Row> rows = IteratorUtils.toList(output.executeAndCollect());
        for (Row value : rows) {
            for (Row exp : expectedData) {
                assert !exp.getField(1).equals(value.getField(0))
                        || (exp.getField(0).equals(value.getField(1)));
            }
        }
    }

    /** test knn model load and transform. */
    @Test
    public void testEstimatorLoadAndSave() throws Exception {
        String path = Files.createTempDirectory("").toString();
        Knn knn =
                new Knn()
                        .setLabelCol("label")
                        .setFeaturesCol("vec")
                        .setK(4)
                        .setPredictionCol("pred");
        knn.save(path);

        Knn loadKnn = Knn.load(path);
        KnnModel knnModel = loadKnn.fit(trainData);
        Table result = knnModel.transform(testData)[0];

        DataStream<Row> output = tEnv.toDataStream(result);

        List<Row> rows = IteratorUtils.toList(output.executeAndCollect());
        for (Row value : rows) {
            for (Row exp : expectedData) {
                assert !exp.getField(1).equals(value.getField(0))
                        || (exp.getField(0).equals(value.getField(1)));
            }
        }
    }

    /** Test knn model load and transform. */
    @Test
    public void testModelLoadAndSave() throws Exception {
        String path = Files.createTempDirectory("").toString();
        Knn knn =
                new Knn()
                        .setLabelCol("label")
                        .setFeaturesCol("vec")
                        .setK(4)
                        .setPredictionCol("pred");
        KnnModel knnModel = knn.fit(trainData);
        knnModel.save(path);
        env.execute();

        KnnModel newModel = KnnModel.load(env, path);
        Table result = newModel.transform(testData)[0];

        DataStream<Row> output = tEnv.toDataStream(result);

        List<Row> rows = IteratorUtils.toList(output.executeAndCollect());
        for (Row value : rows) {
            for (Row exp : expectedData) {
                assert !exp.getField(1).equals(value.getField(0))
                        || (exp.getField(0).equals(value.getField(1)));
            }
        }
    }

    /** Test Param. */
    @Test
    public void testParam() {
        Knn knnOrigin = new Knn();
        assertEquals("label", knnOrigin.getLabelCol());
        assertEquals(10L, knnOrigin.getK().longValue());
        assertEquals("prediction", knnOrigin.getPredictionCol());

        Knn knn =
                new Knn()
                        .setLabelCol("label")
                        .setFeaturesCol("vec")
                        .setK(4)
                        .setPredictionCol("pred");

        assertEquals("vec", knn.getFeaturesCol());
        assertEquals("label", knn.getLabelCol());
        assertEquals(4L, knn.getK().longValue());
        assertEquals("pred", knn.getPredictionCol());
    }

    /** Test model data. */
    @Test
    public void testModelData() throws Exception {
        Knn knn =
                new Knn()
                        .setLabelCol("label")
                        .setFeaturesCol("vec")
                        .setK(4)
                        .setPredictionCol("pred");

        KnnModel knnModel = knn.fit(trainData);
        Table modelData = knnModel.getModelData()[0];

        DataStream<Row> output = tEnv.toDataStream(modelData);

        assertEquals(
                Arrays.asList("DATA", "KNN_LABEL_TYPE"),
                modelData.getResolvedSchema().getColumnNames());

        assertEquals(
                Arrays.asList(DataTypes.STRING(), DataTypes.STRING()),
                modelData.getResolvedSchema().getColumnDataTypes());

        List<Row> modelRows = IteratorUtils.toList(output.executeAndCollect());
        for (int i = 0; i < modelRows.size(); ++i) {
            String data = (String) modelRows.get(0).getField(0);
            FastDistanceMatrixData matrixData = FastDistanceMatrixData.fromString(data);
            assertEquals(2, matrixData.vectors.numRows());
            assertEquals(1, matrixData.label.numRows());
            assertEquals(matrixData.vectors.numCols(), matrixData.label.numCols());
        }
    }
}
