package org.apache.flink.ml.classification;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.connector.source.Source;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.connector.file.src.reader.TextLineInputFormat;
import org.apache.flink.ml.classification.fmclassifier.FmClassifier;
import org.apache.flink.ml.classification.fmclassifier.FmClassifierModel;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.fm.FmModelData;
import org.apache.flink.ml.common.fm.FmModelDataUtil;
import org.apache.flink.ml.common.fm.FmModelServable;
import org.apache.flink.ml.evaluation.binaryclassification.BinaryClassificationEvaluator;
import org.apache.flink.ml.evaluation.binaryclassification.BinaryClassificationEvaluatorParams;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.SparseVectorTypeInfo;
import org.apache.flink.ml.servable.api.DataFrame;
import org.apache.flink.ml.servable.types.BasicType;
import org.apache.flink.ml.servable.types.DataTypes;
import org.apache.flink.ml.util.FileUtils;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
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

/** Tests {@link FmClassifier} and {@link FmClassifierModel}. */
public class FmClassifierTest {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;

    private StreamTableEnvironment tEnv;

    private static final List<Row> trainRows =
            Arrays.asList(
                    Row.of(
                            new SparseVector(5, new int[] {0, 1, 2}, new double[] {1, 2, 1}),
                            0.,
                            1.),
                    Row.of(
                            new SparseVector(5, new int[] {0, 2, 3}, new double[] {2, 3, 1}),
                            0.,
                            2.),
                    Row.of(
                            new SparseVector(5, new int[] {0, 1, 4}, new double[] {2, 3, 2}),
                            0.,
                            3.),
                    Row.of(
                            new SparseVector(5, new int[] {0, 2, 4}, new double[] {1, 3, 3}),
                            0.,
                            4.),
                    Row.of(
                            new SparseVector(5, new int[] {0, 1, 3}, new double[] {1, 2, 1}),
                            0.,
                            5.),
                    Row.of(
                            new SparseVector(5, new int[] {1, 2, 4}, new double[] {11, 32, 12}),
                            1.,
                            1.),
                    Row.of(
                            new SparseVector(5, new int[] {2, 3, 4}, new double[] {12, 14, 12}),
                            1.,
                            2.),
                    Row.of(
                            new SparseVector(5, new int[] {1, 2, 4}, new double[] {13, 12, 12}),
                            1.,
                            3.),
                    Row.of(
                            new SparseVector(5, new int[] {1, 3, 4}, new double[] {14, 42, 13}),
                            1.,
                            4.),
                    Row.of(
                            new SparseVector(5, new int[] {2, 3, 4}, new double[] {15, 43, 21}),
                            1.,
                            5.));

    private static final List<Row> testRows =
            Collections.singletonList(
                    Row.of(Vectors.sparse(4, new int[] {0, 1, 3}, new double[] {1, 2, 1}), 0., 1.));

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
        FmClassifier fmClassifier = new FmClassifier();
        assertEquals("features", fmClassifier.getFeaturesCol());
        assertEquals("label", fmClassifier.getLabelCol());
        assertNull(fmClassifier.getWeightCol());
        assertEquals(20, fmClassifier.getMaxIter());
        assertEquals(1e-6, fmClassifier.getTol(), TOLERANCE);
        assertEquals(0.01, fmClassifier.getLearnRate(), TOLERANCE);
        assertEquals(0.05, fmClassifier.getInitStdEv(), TOLERANCE);
        assertEquals(32, fmClassifier.getGlobalBatchSize());
        assertEquals(0.9, fmClassifier.getBeta1(), TOLERANCE);
        assertEquals(0.999, fmClassifier.getBeta2(), TOLERANCE);
        assertEquals(0.1, fmClassifier.getL1(), TOLERANCE);
        assertEquals(0.1, fmClassifier.getL2(), TOLERANCE);
        assertEquals(0.1, fmClassifier.getAlpha(), TOLERANCE);
        assertEquals(0.1, fmClassifier.getBeta(), TOLERANCE);
        assertEquals("1,1,10", fmClassifier.getDim());
        assertEquals("0.01,0.01,0.01", fmClassifier.getLambda());
        assertEquals(0.9, fmClassifier.getGamma(), TOLERANCE);
        assertEquals("AdaGrad", fmClassifier.getMethod());
        assertEquals("prediction", fmClassifier.getPredictionCol());
        assertEquals("rawPrediction", fmClassifier.getRawPredictionCol());

        fmClassifier
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
                .setPredictionCol("test_predictionCol")
                .setRawPredictionCol("test_rawPredictionCol");

        assertEquals(0.001, fmClassifier.getL1(), TOLERANCE);
        assertEquals(0.002, fmClassifier.getL2(), TOLERANCE);
        assertEquals(0.003, fmClassifier.getAlpha(), TOLERANCE);
        assertEquals(0.004, fmClassifier.getBeta(), TOLERANCE);
        assertEquals(0.005, fmClassifier.getBeta1(), TOLERANCE);
        assertEquals(0.006, fmClassifier.getBeta2(), TOLERANCE);
        assertEquals(0.007, fmClassifier.getGamma(), TOLERANCE);
        assertEquals(0.008, fmClassifier.getInitStdEv(), TOLERANCE);
        assertEquals("1,1,10", fmClassifier.getDim());
        assertEquals("0.01,0.02,0.03", fmClassifier.getLambda());
        assertEquals("Adam", fmClassifier.getMethod());
        assertEquals("test_features", fmClassifier.getFeaturesCol());
        assertEquals("test_label", fmClassifier.getLabelCol());
        assertEquals("test_weight", fmClassifier.getWeightCol());
        assertEquals(1000, fmClassifier.getMaxIter());
        assertEquals(0.001, fmClassifier.getTol(), TOLERANCE);
        assertEquals(0.5, fmClassifier.getLearnRate(), TOLERANCE);
        assertEquals(1000, fmClassifier.getGlobalBatchSize());
        assertEquals("test_predictionCol", fmClassifier.getPredictionCol());
        assertEquals("test_rawPredictionCol", fmClassifier.getRawPredictionCol());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable = trainTable.as("test_features", "test_label", "test_weight");
        FmClassifier fmClassifier =
                new FmClassifier()
                        .setFeaturesCol("test_features")
                        .setLabelCol("test_label")
                        .setWeightCol("test_weight")
                        .setPredictionCol("test_predictionCol")
                        .setRawPredictionCol("test_rawPredictionCol");
        Table output = fmClassifier.fit(tempTable).transform(tempTable)[0];

        assertEquals(
                Arrays.asList(
                        "test_features",
                        "test_label",
                        "test_weight",
                        "test_predictionCol",
                        "test_rawPredictionCol"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndTransform() throws Exception {
        env.setParallelism(4);
        FmClassifier fmClassifier =
                new FmClassifier()
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
                        .setRawPredictionCol("raw")
                        .setPredictionCol("predict");
        FmClassifierModel model = fmClassifier.fit(trainTable);
        Table result = model.transform(testTable)[0];
        Row row = (Row) IteratorUtils.toList(tEnv.toDataStream(result).executeAndCollect()).get(0);
        Assert.assertEquals(
                new DenseVector(new double[] {0.5526637687877789, 0.4473362312122211}),
                row.getFieldAs("raw"));
        Assert.assertEquals(0.0, row.getFieldAs("predict"), 1.0e-8);
    }

    @Test
    public void testMiniBatchOptimizer() throws Exception {
        env.setParallelism(4);
        FmClassifier fmClassifier =
                new FmClassifier()
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
                        .setRawPredictionCol("raw")
                        .setPredictionCol("predict");
        Table middleTable1 = fmClassifier.fit(trainTable).transform(testTable)[0];
        fmClassifier.setMethod("Ftrl").setRawPredictionCol("raw1").setPredictionCol("predict1");
        Table middleTable2 = fmClassifier.fit(trainTable).transform(middleTable1)[0];
        fmClassifier.setMethod("adam").setRawPredictionCol("raw2").setPredictionCol("predict2");
        Table middleTable3 = fmClassifier.fit(trainTable).transform(middleTable2)[0];
        fmClassifier.setMethod("RMSProp").setRawPredictionCol("raw3").setPredictionCol("predict3");
        Table middleTable4 = fmClassifier.fit(trainTable).transform(middleTable3)[0];
        fmClassifier.setMethod("SGD").setRawPredictionCol("raw4").setPredictionCol("predict4");
        Table middleTable5 = fmClassifier.fit(trainTable).transform(middleTable4)[0];
        fmClassifier.setMethod("AdaDelta").setRawPredictionCol("raw5").setPredictionCol("predict5");
        Table middleTable6 = fmClassifier.fit(trainTable).transform(middleTable5)[0];
        fmClassifier.setMethod("Momentum").setRawPredictionCol("raw6").setPredictionCol("predict6");
        Table result = fmClassifier.fit(trainTable).transform(middleTable6)[0];

        Row row = (Row) IteratorUtils.toList(tEnv.toDataStream(result).executeAndCollect()).get(0);
        Assert.assertEquals(
                new DenseVector(new double[] {0.5904058840267317, 0.4095941159732684}),
                row.getFieldAs("raw"));
        Assert.assertEquals(0.0, row.getFieldAs("predict"), 1.0e-8);
        Assert.assertEquals(
                new DenseVector(new double[] {0.729953571655609, 0.27004642834439097}),
                row.getFieldAs("raw1"));
        Assert.assertEquals(0.0, row.getFieldAs("predict1"), 1.0e-8);
        Assert.assertEquals(
                new DenseVector(new double[] {0.6266079134898535, 0.3733920865101466}),
                row.getFieldAs("raw2"));
        Assert.assertEquals(0.0, row.getFieldAs("predict2"), 1.0e-8);
        Assert.assertEquals(
                new DenseVector(new double[] {0.8983325824031893, 0.10166741759681079}),
                row.getFieldAs("raw3"));
        Assert.assertEquals(0.0, row.getFieldAs("predict3"), 1.0e-8);
        Assert.assertEquals(
                new DenseVector(new double[] {0.5407975623491657, 0.45920243765083435}),
                row.getFieldAs("raw4"));
        Assert.assertEquals(0.0, row.getFieldAs("predict4"), 1.0e-8);
        Assert.assertEquals(
                new DenseVector(new double[] {0.5012678027456665, 0.49873219725433343}),
                row.getFieldAs("raw5"));
        Assert.assertEquals(0.0, row.getFieldAs("predict5"), 1.0e-8);
        Assert.assertEquals(
                new DenseVector(new double[] {0.7489696526738796, 0.2510303473261204}),
                row.getFieldAs("raw6"));
        Assert.assertEquals(0.0, row.getFieldAs("predict6"), 1.0e-8);
    }

    @Test
    public void testAvgOptimizer() throws Exception {
        env.setParallelism(4);
        FmClassifier fmClassifier =
                new FmClassifier()
                        .setWeightCol("weight")
                        .setDim("1,1,5")
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setInitStdEv(0.001)
                        .setGlobalBatchSize(4)
                        .setLearnRate(0.01)
                        .setMaxIter(30)
                        .setGamma(0.8)
                        .setMethod("AdaGrad_avg")
                        .setLambda("0.1, 0.1, 0.1")
                        .setRawPredictionCol("raw")
                        .setPredictionCol("predict");
        Table middleTable1 = fmClassifier.fit(trainTable).transform(testTable)[0];
        fmClassifier.setMethod("Ftrl_avg").setRawPredictionCol("raw1").setPredictionCol("predict1");
        Table middleTable2 = fmClassifier.fit(trainTable).transform(middleTable1)[0];
        fmClassifier.setMethod("adam_avg").setRawPredictionCol("raw2").setPredictionCol("predict2");
        Table middleTable3 = fmClassifier.fit(trainTable).transform(middleTable2)[0];
        fmClassifier
                .setMethod("RMSProp_avg")
                .setRawPredictionCol("raw3")
                .setPredictionCol("predict3");
        Table middleTable4 = fmClassifier.fit(trainTable).transform(middleTable3)[0];
        fmClassifier.setMethod("SGD_avg").setRawPredictionCol("raw4").setPredictionCol("predict4");
        Table middleTable5 = fmClassifier.fit(trainTable).transform(middleTable4)[0];
        fmClassifier
                .setMethod("AdaDelta_avg")
                .setRawPredictionCol("raw5")
                .setPredictionCol("predict5");
        Table middleTable6 = fmClassifier.fit(trainTable).transform(middleTable5)[0];
        fmClassifier
                .setMethod("Momentum_avg")
                .setRawPredictionCol("raw6")
                .setPredictionCol("predict6");
        Table result = fmClassifier.fit(trainTable).transform(middleTable6)[0];

        Row row = (Row) IteratorUtils.toList(tEnv.toDataStream(result).executeAndCollect()).get(0);
        Assert.assertEquals(
                new DenseVector(new double[] {0.6446025203361847, 0.35539747966381535}),
                row.getFieldAs("raw"));
        Assert.assertEquals(0.0, row.getFieldAs("predict"), 1.0e-8);
        Assert.assertEquals(
                new DenseVector(new double[] {0.8210159106144985, 0.17898408938550148}),
                row.getFieldAs("raw1"));
        Assert.assertEquals(0.0, row.getFieldAs("predict1"), 1.0e-8);
        Assert.assertEquals(
                new DenseVector(new double[] {0.9166500601744068, 0.08334993982559322}),
                row.getFieldAs("raw2"));
        Assert.assertEquals(0.0, row.getFieldAs("predict2"), 1.0e-8);
        Assert.assertEquals(
                new DenseVector(new double[] {0.8992951746544293, 0.10070482534557065}),
                row.getFieldAs("raw3"));
        Assert.assertEquals(0.0, row.getFieldAs("predict3"), 1.0e-8);
        Assert.assertEquals(
                new DenseVector(new double[] {0.8189154438666246, 0.18108455613337537}),
                row.getFieldAs("raw4"));
        Assert.assertEquals(0.0, row.getFieldAs("predict4"), 1.0e-8);
        Assert.assertEquals(
                new DenseVector(new double[] {0.5232333579591517, 0.47676664204084834}),
                row.getFieldAs("raw5"));
        Assert.assertEquals(0.0, row.getFieldAs("predict5"), 1.0e-8);
        Assert.assertEquals(
                new DenseVector(new double[] {0.9111088482748075, 0.08889115172519249}),
                row.getFieldAs("raw6"));
        Assert.assertEquals(0.0, row.getFieldAs("predict6"), 1.0e-8);
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        env.setParallelism(4);
        FmClassifier fmClassifier =
                new FmClassifier()
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
                        .setRawPredictionCol("raw")
                        .setPredictionCol("predict");
        fmClassifier =
                TestUtils.saveAndReload(
                        tEnv,
                        fmClassifier,
                        tempFolder.newFolder().getAbsolutePath(),
                        FmClassifier::load);
        FmClassifierModel model = fmClassifier.fit(trainTable);
        model =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        FmClassifierModel::load);
        Table result = model.transform(testTable)[0];
        Row row = (Row) IteratorUtils.toList(tEnv.toDataStream(result).executeAndCollect()).get(0);
        Assert.assertEquals(
                new DenseVector(new double[] {0.5526637687877789, 0.4473362312122211}),
                row.getFieldAs("raw"));
        Assert.assertEquals(0.0, row.getFieldAs("predict"), 1.0e-8);
    }

    @Test
    public void testGetModelData() throws Exception {
        env.setParallelism(1);
        FmClassifier fmClassifier =
                new FmClassifier()
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
        Table output = fmClassifier.fit(trainTable).getModelData()[0];
        List<Row> modelRows = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());

        for (Row row : modelRows) {
            FmModelData data = row.getFieldAs(0);
            for (Tuple2<Long, float[]> t2 : data.factors) {
                if (t2.f0 == 0L) {
                    assertArrayEquals(
                            new float[] {-0.07857955f, -0.07887801f, -0.07766823f, -0.07944956f},
                            t2.f1,
                            1.0e-7f);
                } else if (t2.f0 == 1L) {
                    assertArrayEquals(
                            new float[] {0.073209815f, 0.07444382f, 0.07181655f, 0.08008082f},
                            t2.f1,
                            1.0e-7f);
                } else if (t2.f0 == 2L) {
                    assertArrayEquals(
                            new float[] {0.07719299f, 0.0767468f, 0.07575097f, 0.08091559f},
                            t2.f1,
                            1.0e-7f);
                } else if (t2.f0 == 3L) {
                    assertArrayEquals(
                            new float[] {0.06985869f, 0.0714843f, 0.0690421f, 0.08054871f},
                            t2.f1,
                            1.0e-7f);
                } else if (t2.f0 == 4L) {
                    assertArrayEquals(
                            new float[] {0.07603331f, 0.0756547f, 0.07130743f, 0.08055682f},
                            t2.f1,
                            1.0e-7f);
                } else if (t2.f0 == -1L) {
                    assertArrayEquals(new float[] {-0.07710119f, 0.0f, 0.0f, 0.0f}, t2.f1, 1.0e-7f);
                }
            }
        }
    }

    @Test
    public void testSetModelData() throws Exception {
        FmClassifier fmClassifier =
                new FmClassifier()
                        .setWeightCol("weight")
                        .setDim("1,1,5")
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setInitStdEv(0.001)
                        .setLearnRate(0.08)
                        .setMaxIter(30)
                        .setMethod("Adam")
                        .setLambda("0.1, 0.1, 0.1")
                        .setRawPredictionCol("raw")
                        .setPredictionCol("predict");
        FmClassifierModel model = fmClassifier.fit(trainTable);

        FmClassifierModel newModel = new FmClassifierModel();
        ParamUtils.updateExistingParams(newModel, model.getParamMap());
        newModel.setModelData(model.getModelData());
        Table result = newModel.transform(testTable)[0];
        Row row = (Row) IteratorUtils.toList(tEnv.toDataStream(result).executeAndCollect()).get(0);
        Assert.assertEquals(
                new DenseVector(new double[] {0.9043207519264417, 0.09567924807355836}),
                row.getFieldAs("raw"));
        Assert.assertEquals(0.0, row.getFieldAs("predict"), 1.0e-8);
    }

    @Test
    public void testSetModelDataToServable() throws Exception {
        env.setParallelism(3);
        FmClassifier fmClassifier =
                new FmClassifier()
                        .setWeightCol("weight")
                        .setDim("1,1,5")
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setInitStdEv(0.001)
                        .setLearnRate(0.08)
                        .setMaxIter(30)
                        .setMethod("Adam")
                        .setLambda("0.1, 0.1, 0.1");
        FmClassifierModel model = fmClassifier.fit(trainTable);

        byte[] serializedModelData =
                FmModelDataUtil.getModelDataByteStream(model.getModelData()[0])
                        .executeAndCollect()
                        .next();

        FmModelServable servable = new FmModelServable();
        ParamUtils.updateExistingParams(servable, model.getParamMap());
        servable.setModelData(new ByteArrayInputStream(serializedModelData));

        DataFrame output = servable.transform(testDataFrame);
        org.apache.flink.ml.servable.api.Row row = output.collect().get(0);
        Assert.assertEquals(
                new DenseVector(new double[] {0.9043207519264417, 0.09567924807355836}),
                row.get(4));
        Assert.assertEquals(0.0, (double) row.get(3), 1.0e-8);
    }

    @Test
    public void testServableLoadFunction() throws Exception {
        env.setParallelism(1);
        FmClassifier fmClassifier =
                new FmClassifier()
                        .setWeightCol("weight")
                        .setDim("1,1,5")
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setInitStdEv(0.001)
                        .setLearnRate(0.08)
                        .setMaxIter(30)
                        .setMethod("Adam")
                        .setLambda("0.1, 0.1, 0.1");
        FmClassifierModel model = fmClassifier.fit(trainTable);

        String path = tempFolder.newFolder().getAbsolutePath();
        model.save(path);
        env.execute();

        FmModelServable servable = FmClassifierModel.loadServable(path);
        ParamUtils.updateExistingParams(servable, model.getParamMap());

        DataFrame output = servable.transform(testDataFrame);
        org.apache.flink.ml.servable.api.Row row = output.collect().get(0);
        Assert.assertEquals(
                new DenseVector(new double[] {0.9043207519264417, 0.09567924807355836}),
                row.get(4));
        Assert.assertEquals(0.0, (double) row.get(3), 1.0e-8);
    }

    public Table getAdultTable() {
        String path = "/Users/weibo/workspace/data/flink_ml_fm_test_data/adult";
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);
        Source<String, ?, ?> source =
                FileSource.forRecordStreamFormat(
                                new TextLineInputFormat(), FileUtils.getDataPath(path))
                        .build();
        DataStream<Row> ds =
                env.fromSource(source, WatermarkStrategy.noWatermarks(), "movlelens")
                        .map(
                                new MapFunction<String, Row>() {
                                    @Override
                                    public Row map(String s) throws Exception {
                                        String[] contents = s.split(",");
                                        double label = Double.parseDouble(contents[0]);
                                        String[] kvs = contents[1].substring(5).split(" ");
                                        double[] values = new double[kvs.length];
                                        int[] key = new int[kvs.length];
                                        int iter = 0;
                                        for (String e : kvs) {
                                            String[] kv = e.split(":");
                                            key[iter] = Integer.parseInt(kv[0]);
                                            if (iter == 0) {
                                                values[iter++] = Double.parseDouble(kv[1]);
                                            } else {
                                                values[iter++] = Double.parseDouble(kv[1]);
                                            }
                                        }

                                        return Row.of(
                                                new SparseVector(107, key, values), label, 1.0);
                                    }
                                })
                        .returns(
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            SparseVectorTypeInfo.INSTANCE,
                                            Types.DOUBLE,
                                            Types.DOUBLE
                                        },
                                        new String[] {"features", "label", "weight"}));
        return tEnv.fromDataStream(ds);
    }

    public Table getTable(String name) {
        String path = "/Users/weibo/workspace/data/flink_ml_fm_test_data/" + name;
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);
        Source<String, ?, ?> source =
                FileSource.forRecordStreamFormat(
                                new TextLineInputFormat(), FileUtils.getDataPath(path))
                        .build();
        DataStream<Row> ds =
                env.fromSource(source, WatermarkStrategy.noWatermarks(), "movlelens")
                        .map(
                                new RichMapFunction<String, Row>() {
                                    private int max = 0;

                                    @Override
                                    public Row map(String s) throws Exception {
                                        String[] contents = s.split("&");
                                        double label = Integer.parseInt(contents[0]);
                                        String[] kvs = contents[1].split(",");
                                        double[] values = new double[kvs.length];
                                        int[] key = new int[kvs.length];
                                        int iter = 0;
                                        for (String e : kvs) {
                                            String[] kv = e.split(":");
                                            key[iter] = Integer.parseInt(kv[0]);
                                            max = Math.max(max, key[iter]);
                                            values[iter++] = Double.parseDouble(kv[1]);
                                        }

                                        return Row.of(
                                                new SparseVector(1000000, key, values), label, 1.0);
                                    }

                                    @Override
                                    public void close() {
                                        System.out.println(max);
                                    }
                                })
                        .returns(
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            SparseVectorTypeInfo.INSTANCE,
                                            Types.DOUBLE,
                                            Types.DOUBLE
                                        },
                                        new String[] {"features", "label", "weight"}));
        return tEnv.fromDataStream(ds);
    }

    @Test
    public void testAdult() {
        env.setParallelism(2);
        for (int i = 1; i < 2; ++i) {
            FmClassifier fmClassifier =
                    new FmClassifier()
                            .setWeightCol("weight")
                            .setDim("1,1,10")
                            .setFeaturesCol("features")
                            .setLabelCol("label")
                            .setInitStdEv(0.001)
                            .setLearnRate(0.00001)
                            .setGamma(0.7)
                            .setGlobalBatchSize(100000)
                            .setMaxIter(i * 5)
                            .setBeta2(0.5)
                            .setBeta1(0.5)
                            .setAlpha(0.01)
                            .setBeta(0.02)
                            .setL1(0.01)
                            .setL2(0.01)
                            .setMethod("ADADELTA_AVG")
                            .setRawPredictionCol("raw")
                            .setLambda("0.1, 0.1, 0.1");
            Table adult = getAdultTable();
            Table zxzl = getTable("train");
            FmClassifierModel model = fmClassifier.fit(adult);

            Table result = model.transform(adult)[0];

            BinaryClassificationEvaluator eval =
                    new BinaryClassificationEvaluator()
                            .setLabelCol("label")
                            .setRawPredictionCol("raw")
                            .setMetricsNames(
                                    BinaryClassificationEvaluatorParams.AREA_UNDER_PR,
                                    BinaryClassificationEvaluatorParams.KS,
                                    BinaryClassificationEvaluatorParams.AREA_UNDER_ROC);
            Table evalResult = eval.transform(result)[0];

            List<Row> results = IteratorUtils.toList(evalResult.execute().collect());
            for (Row row : results) {
                System.out.println(row);
            }
        }
    }
}
