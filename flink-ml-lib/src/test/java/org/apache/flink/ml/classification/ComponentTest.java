package org.apache.flink.ml.classification;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.classification.fmclassifier.FmClassifier;
import org.apache.flink.ml.classification.fmclassifier.FmClassifierModel;
import org.apache.flink.ml.common.ps.api.MLData;
import org.apache.flink.ml.common.ps.api.MLData.MLDataFunction;
import org.apache.flink.ml.common.ps.api.SinkComponent;
import org.apache.flink.ml.common.ps.api.SourceComponent;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.List;

/** Tests {@link FmClassifier} and {@link FmClassifierModel}. */
public class ComponentTest {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;

    private static final List<Row> trainRows =
            Arrays.asList(
                    Row.of(1, 1., 2., 1),
                    Row.of(2, 1., 2., 1),
                    Row.of(3, 1., 2., 1),
                    Row.of(4, 1., 2., 1),
                    Row.of(5, 1., 2., 1),
                    Row.of(6, 1., 1., 1),
                    Row.of(7, 1., 1., 1),
                    Row.of(8, 1., 1., 1),
                    Row.of(9, 1., 1., 1),
                    Row.of(10, 1., 1., 1));

    private Table trainTable;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        env.getConfig().enableGenericTypes();
        env.getCheckpointConfig().disableCheckpointing();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        trainTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                trainRows,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            Types.INT, Types.DOUBLE, Types.DOUBLE, Types.INT
                                        },
                                        new String[] {"features", "label", "weight", "append"})));
    }

    @Test
    public void testSource() throws Exception {
        MLData mlData =
                new SourceComponent()
                        .filePath("src/test/resources/source.txt")
                        .delimiter(",")
                        .output("output")
                        .schema("a int, b double, c string")
                        .source();
        new MLDataFunction("shuffle").apply(mlData);

        new SinkComponent().filePath("/tmp/sink.csv").input("output").sink(mlData);
        List<?> rows = mlData.executeAndCollect(0);
        for (Object row : rows) {
            System.out.println(row);
        }
    }
}
