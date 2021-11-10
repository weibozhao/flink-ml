package org.apache.flink.ml.algo.batch.knn;

import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.iteration.config.IterationOptions;
import org.apache.flink.ml.api.core.Pipeline;
import org.apache.flink.ml.api.core.Stage;
import org.apache.flink.ml.common.BatchOperator;
import org.apache.flink.ml.common.MLEnvironmentFactory;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.scala.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class KnnBatchOpTest {
    private BatchOperator<?> getSourceOp(List<Row> rows) {
        DataStream<Row> dataStream =
                MLEnvironmentFactory.getDefault()
                        .getStreamExecutionEnvironment()
                        .fromCollection(
                                rows,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            Types.INT, Types.STRING, Types.DOUBLE
                                        },
                                        new String[] {"re", "vec", "label"}));

        Table out =
                MLEnvironmentFactory.getDefault()
                        .getStreamTableEnvironment()
                        .fromDataStream(dataStream);
        return new TableSourceBatchOp(out);
    }

    private Table getTable(List<Row> rows) {
        DataStream<Row> dataStream =
                MLEnvironmentFactory.getDefault()
                        .getStreamExecutionEnvironment()
                        .fromCollection(
                                rows,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            Types.INT, Types.STRING, Types.DOUBLE
                                        },
                                        new String[] {"re", "vec", "label"}));

        Table out =
                MLEnvironmentFactory.getDefault()
                        .getStreamTableEnvironment()
                        .fromDataStream(dataStream);
        return out;
    }

    @Rule public TemporaryFolder tempFolder = new TemporaryFolder();

    @Test
    public void testKnnTrainBatchOp() throws Exception {

        StreamExecutionEnvironment.setDefaultLocalParallelism(1);
        org.apache.flink.streaming.api.environment.StreamExecutionEnvironment env =
                MLEnvironmentFactory.getDefault().getStreamExecutionEnvironment();
        env.setRuntimeMode(RuntimeExecutionMode.BATCH);
        Configuration configuration = new Configuration();
        configuration.set(RestOptions.PORT, 18082);
        configuration.set(
                IterationOptions.DATA_CACHE_PATH,
                "file://" + tempFolder.newFolder().getAbsolutePath());
        configuration.set(
                ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env.getConfig().setGlobalJobParameters(configuration);

        List<Row> rows =
                Arrays.asList(
                        Row.of(1, "1 2 3 4", 1.),
                        Row.of(1, "1 2 3 4.2", 2.),
                        Row.of(1, "1 2 3 4.3", 3.),
                        Row.of(1, "1 2 3 4.4", 4.),
                        Row.of(1, "1 2 3 4.5", 5.),
                        Row.of(1, "3 2 3 4.6", 6.),
                        Row.of(1, "1 2 3 4.7", 7.),
                        Row.of(1, "3 2 3 4.9", 8.));

        BatchOperator source = getSourceOp(rows);
        BatchOperator<?> knn =
                new KnnTrainBatchOp().setLabelCol("label").setVectorCol("vec").linkFrom(source);

        BatchOperator result =
                new KnnPredictBatchOp(null)
                        .setK(2)
                        .setReservedCols("re", "label")
                        .setPredictionCol("pred")
                        .setPredictionDetailCol("detail")
                        .linkFrom(source, knn);

        MLEnvironmentFactory.getDefault()
                .getStreamTableEnvironment()
                .toDataStream(result.getOutput())
                .addSink(
                        new SinkFunction<Row>() {
                            @Override
                            public void invoke(Row value, Context context) throws Exception {
                                System.out.println("[Output]: " + value.toString());
                            }
                        });
        MLEnvironmentFactory.getDefault().getStreamExecutionEnvironment().execute();
    }

    @Test
    public void testKnnPipeline() throws Exception {

        StreamExecutionEnvironment.setDefaultLocalParallelism(4);
        org.apache.flink.streaming.api.environment.StreamExecutionEnvironment env =
                MLEnvironmentFactory.getDefault().getStreamExecutionEnvironment();
        env.setRuntimeMode(RuntimeExecutionMode.BATCH);
        Configuration configuration = new Configuration();
        configuration.set(RestOptions.PORT, 18082);
        configuration.set(
                IterationOptions.DATA_CACHE_PATH,
                "file://" + tempFolder.newFolder().getAbsolutePath());
        configuration.set(
                ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env.getConfig().setGlobalJobParameters(configuration);

        List<Row> rows =
                Arrays.asList(
                        Row.of(1, "1 2 3 4", 1.),
                        Row.of(1, "1 2 3 4.2", 2.),
                        Row.of(1, "1 2 3 4.3", 3.),
                        Row.of(1, "1 2 3 4.4", 4.),
                        Row.of(1, "1 2 3 4.5", 5.),
                        Row.of(1, "3 2 3 4.6", 6.),
                        Row.of(1, "1 2 3 4.7", 7.),
                        Row.of(1, "3 2 3 4.9", 8.));

        Table source = getTable(rows);
        KnnClassifier knn =
                new KnnClassifier()
                        .setLabelCol("label")
                        .setVectorCol("vec")
                        .setReservedCols("label")
                        .setK(2)
                        .setPredictionCol("pred")
                        .setPredictionDetailCol("detail");
        List<Stage<?>> stages = new ArrayList<>();
        stages.add(knn);

        Pipeline pipe = new Pipeline(stages);

        // KnnClassificationModel knnModel = knn.fit(source);
        // Table result = knnModel.transform(source)[0];
        Table result = pipe.fit(source).transform(source)[0];

        MLEnvironmentFactory.getDefault()
                .getStreamTableEnvironment()
                .toDataStream(result)
                .addSink(
                        new SinkFunction<Row>() {
                            @Override
                            public void invoke(Row value, Context context) throws Exception {
                                System.out.println("[Output]: " + value.toString());
                            }
                        });
        MLEnvironmentFactory.getDefault().getStreamExecutionEnvironment().execute();
    }

    public final class TableSourceBatchOp extends BatchOperator<TableSourceBatchOp> {

        public TableSourceBatchOp(Table table) {
            super(null);
            Preconditions.checkArgument(table != null, "The source table cannot be null.");
            this.setOutput(table);
        }

        @Override
        public Table[] transform(Table... inputs) {
            throw new UnsupportedOperationException(
                    "Table source operator should not have any upstream to link from.");
        }

        @Override
        public void save(String path) throws IOException {
            ReadWriteUtils.saveMetadata(this, path);
        }

        public TableSourceBatchOp load(String path) throws IOException {
            return ReadWriteUtils.loadStageParam(path);
        }
    }
}
