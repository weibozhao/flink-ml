package org.apache.flink.ml.classification.knn;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.MapPartitionFunctionWrapper;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.table.types.DataType;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * KNN is to classify unlabeled observations by assigning them to the class of the most similar
 * labeled examples.
 */
public class Knn implements Estimator<Knn, KnnModel>, KnnParams<Knn> {

    protected Map<Param<?>, Object> params = new HashMap<>();

    /** constructor. */
    public Knn() {
        ParamUtils.initializeMapWithDefaultValues(params, this);
    }

    /**
     * constructor.
     *
     * @param params parameters for algorithm.
     */
    public Knn(Map<Param<?>, Object> params) {
        this.params = params;
    }

    /**
     * @param inputs a list of tables
     * @return knn classification model.
     */
    @Override
    public KnnModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        ResolvedSchema schema = inputs[0].getResolvedSchema();
        String[] colNames = schema.getColumnNames().toArray(new String[0]);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> input = tEnv.toDataStream(inputs[0]);

        String labelCol = getLabelCol();
        String vecCol = getFeaturesCol();

        DataStream<Row> trainData =
                input.map(
                        (MapFunction<Row, Row>)
                                value -> {
                                    Object label = value.getField(labelCol);
                                    DenseVector vec = (DenseVector) value.getField(vecCol);
                                    return Row.of(label, vec);
                                });

        DataType idType = schema.getColumnDataTypes().get(findColIndex(colNames, labelCol));
        DataStream<Row> model = buildModel(trainData, idType);
        KnnModel knnModel =
                new KnnModel()
                        .setFeaturesCol(getFeaturesCol())
                        .setK(getK())
                        .setPredictionCol(getPredictionCol());
        knnModel.setModelData(tEnv.fromDataStream(model, KnnModelData.getModelSchema(idType)));
        return knnModel;
    }

    /**
     * build knn model.
     *
     * @param dataStream input data.
     * @return stream format model.
     */
    private static DataStream<Row> buildModel(DataStream<Row> dataStream, final DataType idType) {
        FastDistance fastDistance = new FastDistance();
        Schema schema = KnnModelData.getModelSchema(idType);
        return dataStream.transform(
                "build index",
                TableUtils.getRowTypeInfo(schema),
                new MapPartitionFunctionWrapper<>(
                        new RichMapPartitionFunction<Row, Row>() {
                            @Override
                            public void mapPartition(Iterable<Row> values, Collector<Row> out) {
                                List<FastDistanceMatrixData> list =
                                        fastDistance.prepareMatrixData(values, 1, 0);
                                for (FastDistanceMatrixData fastDistanceData : list) {
                                    Row row = new Row(2);
                                    row.setField(0, fastDistanceData.toString());
                                    out.collect(row);
                                }
                            }
                        }));
    }

    /**
     * Find the index of <code>targetCol</code> in string array <code>tableCols</code>. It will
     * ignore the case of the tableCols.
     *
     * @param tableCols a string array among which to find the targetCol.
     * @param targetCol the targetCol to find.
     * @return the index of the targetCol, if not found, returns -1.
     */
    private int findColIndex(String[] tableCols, String targetCol) {
        Preconditions.checkNotNull(targetCol, "targetCol is null!");
        for (int i = 0; i < tableCols.length; i++) {
            if (targetCol.equalsIgnoreCase(tableCols[i])) {
                return i;
            }
        }
        return -1;
    }

    /** @return parameters of this algorithm. */
    @Override
    public Map<Param<?>, Object> getParamMap() {
        return this.params;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static Knn load(String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }
}
