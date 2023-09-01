package org.apache.flink.ml.regression;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.common.fm.FmModelData;
import org.apache.flink.ml.common.fm.FmModelDataUtil;
import org.apache.flink.ml.common.fm.FmModelServable;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.SparseVectorTypeInfo;
import org.apache.flink.ml.regression.fmregressor.FmRegressor;
import org.apache.flink.ml.regression.fmregressor.FmRegressorModel;
import org.apache.flink.ml.servable.api.DataFrame;
import org.apache.flink.ml.servable.types.BasicType;
import org.apache.flink.ml.servable.types.DataTypes;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/** Tests {@link FmRegressor} and {@link FmRegressorModel}. */
public class FmRegressorTest {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;

    private StreamTableEnvironment tEnv;

    private static final List<Row> trainRows =
            Arrays.asList(
                    Row.of(
                            new SparseVector(5, new int[] {0, 1, 2}, new double[] {1, 2, 1}),
                            1.,
                            1.),
                    Row.of(
                            new SparseVector(5, new int[] {0, 2, 3}, new double[] {2, 3, 1}),
                            3.,
                            2.),
                    Row.of(
                            new SparseVector(5, new int[] {0, 1, 4}, new double[] {2, 3, 2}),
                            4.,
                            3.),
                    Row.of(
                            new SparseVector(5, new int[] {0, 2, 4}, new double[] {1, 3, 3}),
                            5.,
                            4.),
                    Row.of(
                            new SparseVector(5, new int[] {0, 1, 3}, new double[] {1, 2, 1}),
                            6.,
                            5.),
                    Row.of(
                            new SparseVector(5, new int[] {1, 2, 4}, new double[] {11, 32, 12}),
                            11.,
                            1.),
                    Row.of(
                            new SparseVector(5, new int[] {2, 3, 4}, new double[] {12, 14, 12}),
                            12.,
                            2.),
                    Row.of(
                            new SparseVector(5, new int[] {1, 2, 4}, new double[] {13, 12, 12}),
                            13.,
                            3.),
                    Row.of(
                            new SparseVector(5, new int[] {1, 3, 4}, new double[] {14, 42, 13}),
                            14.,
                            4.),
                    Row.of(
                            new SparseVector(5, new int[] {2, 3, 4}, new double[] {15, 43, 21}),
                            15.,
                            5.));

    private static final List<Row> testRows =
            Collections.singletonList(
                    Row.of(Vectors.sparse(4, new int[] {0, 1, 2}, new double[] {1, 2, 1}), 1., 1.));

    private Table trainTable;
    private Table testTable;
    private DataFrame testDataFrame;
    private static final double TOLERANCE = 1e-7;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        env.getConfig().enableGenericTypes();
        env.getCheckpointConfig().disableCheckpointing();
        tEnv = StreamTableEnvironment.create(env);

        trainTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                trainRows,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            SparseVectorTypeInfo.INSTANCE,
                                            Types.DOUBLE,
                                            Types.DOUBLE
                                        },
                                        new String[] {"features", "label", "weight"})));
        testTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                testRows,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            SparseVectorTypeInfo.INSTANCE,
                                            Types.DOUBLE,
                                            Types.DOUBLE
                                        },
                                        new String[] {"features", "label", "weight"})));

        testDataFrame =
                TestUtils.constructDataFrame(
                        new ArrayList<>(Arrays.asList("features", "label", "weight")),
                        new ArrayList<>(
                                Arrays.asList(
                                        DataTypes.vectorType(BasicType.DOUBLE),
                                        DataTypes.DOUBLE,
                                        DataTypes.DOUBLE)),
                        testRows);
    }

    @Test
    public void testParam() {
        FmRegressor fmRegressor = new FmRegressor();
        assertEquals("features", fmRegressor.getFeaturesCol());
        assertEquals("label", fmRegressor.getLabelCol());
        assertNull(fmRegressor.getWeightCol());
        assertEquals(20, fmRegressor.getMaxIter());
        assertEquals(1e-6, fmRegressor.getTol(), TOLERANCE);
        assertEquals(0.01, fmRegressor.getLearnRate(), TOLERANCE);
        assertEquals(0.05, fmRegressor.getInitStdEv(), TOLERANCE);
        assertEquals(32, fmRegressor.getGlobalBatchSize());
        assertEquals(0.9, fmRegressor.getBeta1(), TOLERANCE);
        assertEquals(0.999, fmRegressor.getBeta2(), TOLERANCE);
        assertEquals(0.1, fmRegressor.getL1(), TOLERANCE);
        assertEquals(0.1, fmRegressor.getL2(), TOLERANCE);
        assertEquals(0.1, fmRegressor.getAlpha(), TOLERANCE);
        assertEquals(0.1, fmRegressor.getBeta(), TOLERANCE);
        assertEquals("1,1,10", fmRegressor.getDim());
        assertEquals("0.01,0.01,0.01", fmRegressor.getLambda());
        assertEquals(0.9, fmRegressor.getGamma(), TOLERANCE);
        assertEquals("AdaGrad", fmRegressor.getMethod());
        assertEquals("prediction", fmRegressor.getPredictionCol());

        fmRegressor
                .setL1(0.001)
                .setL2(0.002)
                .setAlpha(0.003)
                .setBeta(0.004)
                .setBeta1(0.005)
                .setBeta2(0.006)
                .setGamma(0.007)
                .setDim("1,1,10")
                .setLambda("0.01,0.02,0.03")
                .setMethod("Adam")
                .setInitStdEv(0.008)
                .setFeaturesCol("test_features")
                .setLabelCol("test_label")
                .setWeightCol("test_weight")
                .setMaxIter(1000)
                .setTol(0.001)
                .setLearnRate(0.5)
                .setGlobalBatchSize(1000)
                .setPredictionCol("test_predictionCol");

        assertEquals(0.001, fmRegressor.getL1(), TOLERANCE);
        assertEquals(0.002, fmRegressor.getL2(), TOLERANCE);
        assertEquals(0.003, fmRegressor.getAlpha(), TOLERANCE);
        assertEquals(0.004, fmRegressor.getBeta(), TOLERANCE);
        assertEquals(0.005, fmRegressor.getBeta1(), TOLERANCE);
        assertEquals(0.006, fmRegressor.getBeta2(), TOLERANCE);
        assertEquals(0.007, fmRegressor.getGamma(), TOLERANCE);
        assertEquals(0.008, fmRegressor.getInitStdEv(), TOLERANCE);
        assertEquals("1,1,10", fmRegressor.getDim());
        assertEquals("0.01,0.02,0.03", fmRegressor.getLambda());
        assertEquals("Adam", fmRegressor.getMethod());
        assertEquals("test_features", fmRegressor.getFeaturesCol());
        assertEquals("test_label", fmRegressor.getLabelCol());
        assertEquals("test_weight", fmRegressor.getWeightCol());
        assertEquals(1000, fmRegressor.getMaxIter());
        assertEquals(0.001, fmRegressor.getTol(), TOLERANCE);
        assertEquals(0.5, fmRegressor.getLearnRate(), TOLERANCE);
        assertEquals(1000, fmRegressor.getGlobalBatchSize());
        assertEquals("test_predictionCol", fmRegressor.getPredictionCol());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable = trainTable.as("test_features", "test_label", "test_weight");
        FmRegressor fmRegressor =
                new FmRegressor()
                        .setFeaturesCol("test_features")
                        .setLabelCol("test_label")
                        .setWeightCol("test_weight")
                        .setPredictionCol("test_predictionCol");
        Table output = fmRegressor.fit(tempTable).transform(tempTable)[0];

        assertEquals(
                Arrays.asList("test_features", "test_label", "test_weight", "test_predictionCol"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndTransform() throws Exception {
        env.setParallelism(4);
        FmRegressor fmRegressor =
                new FmRegressor()
                        .setWeightCol("weight")
                        .setDim("1,1,5")
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setInitStdEv(0.001)
                        .setGlobalBatchSize(32)
                        .setLearnRate(0.08)
                        .setMaxIter(2)
                        .setMethod("AdaGrad")
                        .setLambda("0.1, 0.1, 0.1")
                        .setPredictionCol("predict");
        FmRegressorModel model = fmRegressor.fit(trainTable);
        Table result = model.transform(testTable)[0];
        Row row = (Row) IteratorUtils.toList(tEnv.toDataStream(result).executeAndCollect()).get(0);
        Assert.assertEquals(0.18583209812641144, row.getFieldAs("predict"), 1.0e-8);
    }

    @Test
    public void testOptimizer() throws Exception {
        env.setParallelism(4);
        FmRegressor fmRegressor =
                new FmRegressor()
                        .setWeightCol("weight")
                        .setDim("1,1,5")
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setInitStdEv(0.001)
                        .setGlobalBatchSize(4)
                        .setLearnRate(0.01)
                        .setMaxIter(30)
                        .setGamma(0.8)
                        .setMethod("AdaGrad")
                        .setLambda("0.1, 0.1, 0.1")
                        .setPredictionCol("predict");
        Table middleTable1 = fmRegressor.fit(trainTable).transform(testTable)[0];
        fmRegressor.setMethod("Ftrl").setPredictionCol("predict1");
        Table middleTable2 = fmRegressor.fit(trainTable).transform(middleTable1)[0];
        fmRegressor.setMethod("adam").setPredictionCol("predict2");
        Table middleTable3 = fmRegressor.fit(trainTable).transform(middleTable2)[0];
        fmRegressor.setMethod("RMSProp").setPredictionCol("predict3");
        Table middleTable4 = fmRegressor.fit(trainTable).transform(middleTable3)[0];
        fmRegressor.setMethod("SGD").setLearnRate(0.0001).setPredictionCol("predict4");
        Table middleTable5 = fmRegressor.fit(trainTable).transform(middleTable4)[0];
        fmRegressor.setMethod("AdaDelta").setPredictionCol("predict5");
        Table middleTable6 = fmRegressor.fit(trainTable).transform(middleTable5)[0];
        fmRegressor.setMethod("Momentum").setPredictionCol("predict6");
        Table result = fmRegressor.fit(trainTable).transform(middleTable6)[0];

        Row row = (Row) IteratorUtils.toList(tEnv.toDataStream(result).executeAndCollect()).get(0);

        Assert.assertEquals(0.5501255393028259, row.getFieldAs("predict"), 1.0e-8);
        Assert.assertEquals(2.2269744873046875, row.getFieldAs("predict1"), 1.0e-8);
        Assert.assertEquals(1.5114543437957764, row.getFieldAs("predict2"), 1.0e-8);
        Assert.assertEquals(1.706116795539856, row.getFieldAs("predict3"), 1.0e-8);
        Assert.assertEquals(0.5043978095054626, row.getFieldAs("predict4"), 1.0e-8);
        Assert.assertEquals(0.1197570413351059, row.getFieldAs("predict5"), 1.0e-8);
        Assert.assertEquals(0.9788283109664917, row.getFieldAs("predict6"), 1.0e-8);
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        env.setParallelism(4);
        FmRegressor fmRegressor =
                new FmRegressor()
                        .setWeightCol("weight")
                        .setDim("1,1,5")
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setInitStdEv(0.001)
                        .setGlobalBatchSize(32)
                        .setLearnRate(0.08)
                        .setMaxIter(2)
                        .setMethod("AdaGrad")
                        .setLambda("0.1, 0.1, 0.1")
                        .setPredictionCol("predict");
        fmRegressor =
                TestUtils.saveAndReload(
                        tEnv,
                        fmRegressor,
                        tempFolder.newFolder().getAbsolutePath(),
                        FmRegressor::load);
        FmRegressorModel model = fmRegressor.fit(trainTable);
        model =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        FmRegressorModel::load);
        Table result = model.transform(testTable)[0];
        Row row = (Row) IteratorUtils.toList(tEnv.toDataStream(result).executeAndCollect()).get(0);
        Assert.assertEquals(0.18583209812641144, row.getFieldAs("predict"), 1.0e-8);
    }

    @Test
    public void testGetModelData() throws Exception {
        env.setParallelism(1);
        FmRegressor fmRegressor =
                new FmRegressor()
                        .setWeightCol("weight")
                        .setDim("1,1,3")
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setInitStdEv(0.001)
                        .setGlobalBatchSize(8)
                        .setLearnRate(0.08)
                        .setMaxIter(2)
                        .setMethod("AdaGrad")
                        .setLambda("0.1, 0.1, 0.1");
        Table output = fmRegressor.fit(trainTable).getModelData()[0];
        List<Row> modelRows = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());

        for (Row row : modelRows) {
            FmModelData data = row.getFieldAs(0);
            for (Tuple2<Long, float[]> t2 : data.factors) {
                if (t2.f0 == 0L) {
                    assertArrayEquals(
                            new float[] {0.12732524f, 0.1256474f, 0.12099945f, 0.13577887f},
                            t2.f1,
                            1.0e-7f);
                } else if (t2.f0 == 1L) {
                    assertArrayEquals(
                            new float[] {0.0015494842f, 9.350925E-4f, 6.6113775E-4f, 0.076575495f},
                            t2.f1,
                            1.0e-7f);
                } else if (t2.f0 == 2L) {
                    assertArrayEquals(
                            new float[] {0.001317327f, 0.001286447f, 8.51548E-4f, 0.0764627f},
                            t2.f1,
                            1.0e-7f);
                } else if (t2.f0 == 3L) {
                    assertArrayEquals(
                            new float[] {0.0012999837f, 6.879574E-4f, 5.688492E-4f, 0.047855932f},
                            t2.f1,
                            1.0e-7f);
                } else if (t2.f0 == 4L) {
                    assertArrayEquals(
                            new float[] {0.0013542969f, 0.0012655297f, 0.001152972f, 0.06740911f},
                            t2.f1,
                            1.0e-7f);
                } else if (t2.f0 == -1L) {
                    assertArrayEquals(new float[] {0.075566016f, 0.0f, 0.0f, 0.0f}, t2.f1, 1.0e-7f);
                }
            }
        }
    }

    @Test
    public void testSetModelData() throws Exception {
        FmRegressor fmRegressor =
                new FmRegressor()
                        .setWeightCol("weight")
                        .setDim("1,1,5")
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setInitStdEv(0.001)
                        .setLearnRate(0.08)
                        .setMaxIter(30)
                        .setMethod("Adam")
                        .setLambda("0.1, 0.1, 0.1")
                        .setPredictionCol("predict");
        FmRegressorModel model = fmRegressor.fit(trainTable);

        FmRegressorModel newModel = new FmRegressorModel();
        ParamUtils.updateExistingParams(newModel, model.getParamMap());
        newModel.setModelData(model.getModelData());
        Table result = newModel.transform(testTable)[0];
        Row row = (Row) IteratorUtils.toList(tEnv.toDataStream(result).executeAndCollect()).get(0);
        Assert.assertEquals(3.5184452533721924, row.getFieldAs("predict"), 1.0e-8);
    }

    @Test
    public void testSetModelDataToServable() throws Exception {
        env.setParallelism(3);
        FmRegressor fmRegressor =
                new FmRegressor()
                        .setWeightCol("weight")
                        .setDim("1,1,5")
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setInitStdEv(0.001)
                        .setLearnRate(0.08)
                        .setMaxIter(30)
                        .setMethod("Adam")
                        .setLambda("0.1, 0.1, 0.1");
        FmRegressorModel model = fmRegressor.fit(trainTable);

        byte[] serializedModelData =
                FmModelDataUtil.getModelDataByteStream(model.getModelData()[0])
                        .executeAndCollect()
                        .next();

        FmModelServable servable = new FmModelServable();
        ParamUtils.updateExistingParams(servable, model.getParamMap());
        servable.setModelData(new ByteArrayInputStream(serializedModelData));

        DataFrame output = servable.transform(testDataFrame);
        org.apache.flink.ml.servable.api.Row row = output.collect().get(0);
        Assert.assertEquals(3.5184452533721924, (double) row.get(3), 1.0e-8);
    }

    @Test
    public void testServableLoadFunction() throws Exception {
        env.setParallelism(1);
        FmRegressor fmRegressor =
                new FmRegressor()
                        .setWeightCol("weight")
                        .setDim("1,1,5")
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setInitStdEv(0.001)
                        .setLearnRate(0.08)
                        .setMaxIter(30)
                        .setMethod("Adam")
                        .setLambda("0.1, 0.1, 0.1");
        FmRegressorModel model = fmRegressor.fit(trainTable);

        String path = tempFolder.newFolder().getAbsolutePath();
        model.save(path);
        env.execute();

        FmModelServable servable = FmRegressorModel.loadServable(path);
        ParamUtils.updateExistingParams(servable, model.getParamMap());

        DataFrame output = servable.transform(testDataFrame);
        org.apache.flink.ml.servable.api.Row row = output.collect().get(0);
        Assert.assertEquals(3.5184452533721924, (double) row.get(3), 1.0e-8);
    }
}
