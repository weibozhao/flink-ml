package org.apache.flink.ml.classification.knn;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.AbstractRichFunction;
import org.apache.flink.api.connector.source.Source;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.connector.file.sink.FileSink;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.core.fs.Path;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.filesystem.bucketassigners.BasePathBucketAssigner;
import org.apache.flink.streaming.api.functions.sink.filesystem.rollingpolicies.OnCheckpointRollingPolicy;
import org.apache.flink.streaming.api.operators.AbstractUdfStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.table.types.DataType;
import org.apache.flink.table.types.logical.utils.LogicalTypeParser;
import org.apache.flink.table.types.utils.LogicalTypeDataTypeConverter;
import org.apache.flink.types.Row;

import org.apache.flink.shaded.curator4.com.google.common.collect.ImmutableMap;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.core.JsonProcessingException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.TreeMap;
import java.util.function.Function;

/** Knn classification model fitted by estimator. */
public class KnnModel implements Model<KnnModel>, KnnModelParams<KnnModel> {
    protected Map<Param<?>, Object> params = new HashMap<>();
    private Table[] modelData;

    /** constructor. */
    public KnnModel() {
        ParamUtils.initializeMapWithDefaultValues(params, this);
    }

    /**
     * Set model data for knn prediction.
     *
     * @param modelData knn model.
     * @return knn model.
     */
    @Override
    public KnnModel setModelData(Table... modelData) {
        this.modelData = modelData;
        return this;
    }

    /**
     * get model data.
     *
     * @return list of tables.
     */
    @Override
    public Table[] getModelData() {
        return modelData;
    }

    /**
     * @param inputs a list of tables.
     * @return result.
     */
    @Override
    public Table[] transform(Table... inputs) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> input = tEnv.toDataStream(inputs[0]);
        DataStream<Row> model = tEnv.toDataStream(modelData[0]);
        final String broadcastKey = "broadcastModelKey";
        Map<String, DataStream<?>> broadcastMap = new HashMap<>(1);
        broadcastMap.put(broadcastKey, model);
        ResolvedSchema modelSchema = modelData[0].getResolvedSchema();

        DataType idType =
                modelSchema.getColumnDataTypes().get(modelSchema.getColumnNames().size() - 1);
        String[] resultCols = new String[] {(String) params.get(KnnModelParams.PREDICTION_COL)};
        DataType[] resultTypes = new DataType[] {idType};

        ResolvedSchema outputSchema =
                TableUtils.getOutputSchema(inputs[0].getResolvedSchema(), resultCols, resultTypes);

        Function<List<DataStream<?>>, DataStream<Row>> function =
                dataStreams -> {
                    DataStream stream = dataStreams.get(0);
                    return stream.transform(
                            "mapFunc",
                            TableUtils.getRowTypeInfo(outputSchema),
                            new PredictOperator(
                                    inputs[0].getResolvedSchema(),
                                    broadcastKey,
                                    getK(),
                                    getFeaturesCol(),
                                    idType));
                };

        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(input), broadcastMap, function);
        return new Table[] {tEnv.fromDataStream(output)};
    }

    /**
     * this operator use mapper to load the model data and do the prediction. if you want to write a
     * prediction operator, you need implement a special mapper for this operator.
     */
    private static class PredictOperator
            extends AbstractUdfStreamOperator<Row, AbstractRichFunction>
            implements OneInputStreamOperator<Row, Row> {

        private boolean firstEle = true;
        private final String[] reservedCols;
        private final String vectorCol;
        private final DataType idType;
        private transient KnnModelData modelData;
        private final Integer topN;
        private final String broadcastKey;

        public PredictOperator(
                ResolvedSchema dataSchema,
                String broadcastKey,
                int k,
                String vectorCol,
                DataType idType) {
            super(new AbstractRichFunction() {});
            reservedCols = dataSchema.getColumnNames().toArray(new String[0]);
            this.topN = k;
            this.broadcastKey = broadcastKey;
            this.vectorCol = vectorCol;
            this.idType = idType;
        }

        @Override
        public void processElement(StreamRecord<Row> streamRecord) throws Exception {
            Row value = streamRecord.getValue();
            output.collect(new StreamRecord<>(map(value)));
        }

        public Row map(Row row) throws Exception {
            if (firstEle) {
                loadModel(userFunction.getRuntimeContext().getBroadcastVariable(broadcastKey));
                firstEle = false;
            }
            DenseVector vector = (DenseVector) row.getField(vectorCol);
            String s = findNeighbor(vector, topN, modelData).toLowerCase();
            Row ret = new Row(reservedCols.length + 1);
            for (int i = 0; i < reservedCols.length; ++i) {
                ret.setField(i, row.getField(reservedCols[i]));
            }

            Tuple2<Object, String> tuple2 = getResultFormat(extractObject(s, idType));
            ret.setField(reservedCols.length, tuple2.f0);
            return ret;
        }

        /**
         * find the nearest topN neighbors from whole nodes.
         *
         * @param input input node.
         * @param topN top N.
         * @return neighbor.
         */
        private String findNeighbor(Object input, Integer topN, KnnModelData modelData)
                throws JsonProcessingException {
            PriorityQueue<Tuple2<Double, Object>> priorityQueue =
                    new PriorityQueue<>(modelData.getQueueComparator());
            search(input, topN, priorityQueue, modelData);
            List<Object> items = new ArrayList<>();
            List<Double> metrics = new ArrayList<>();
            while (!priorityQueue.isEmpty()) {
                Tuple2<Double, Object> result = priorityQueue.poll();
                items.add(result.f1);
                metrics.add(result.f0);
            }
            Collections.reverse(items);
            Collections.reverse(metrics);
            priorityQueue.clear();
            return serializeResult(items, ImmutableMap.of("METRIC", metrics));
        }

        /**
         * serialize result to json format.
         *
         * @param objectValue the nearest nodes found.
         * @param others the metric of nodes.
         * @return serialize result.
         */
        private String serializeResult(List<Object> objectValue, Map<String, List<Double>> others)
                throws JsonProcessingException {
            final String id = "ID";
            Map<String, String> result =
                    new TreeMap<>(
                            (o1, o2) -> {
                                if (id.equals(o1) && id.equals(o2)) {
                                    return 0;
                                } else if (id.equals(o1)) {
                                    return -1;
                                } else if (id.equals(o2)) {
                                    return 1;
                                }
                                return o1.compareTo(o2);
                            });

            result.put(id, ReadWriteUtils.OBJECT_MAPPER.writeValueAsString(objectValue));

            if (others != null) {
                for (Map.Entry<String, List<Double>> other : others.entrySet()) {
                    result.put(
                            other.getKey(),
                            ReadWriteUtils.OBJECT_MAPPER.writeValueAsString(other.getValue()));
                }
            }
            return ReadWriteUtils.OBJECT_MAPPER.writeValueAsString(result);
        }

        /**
         * @param input input node.
         * @param topN top N.
         * @param priorityQueue priority queue.
         */
        private void search(
                Object input,
                Integer topN,
                PriorityQueue<Tuple2<Double, Object>> priorityQueue,
                KnnModelData modelData) {
            Object sample = prepareSample(input, modelData);
            Tuple2<Double, Object> head = null;
            for (int i = 0; i < modelData.getLength(); i++) {
                ArrayList<Tuple2<Double, Object>> values = computeDistance(sample, i);
                if (null == values || values.size() == 0) {
                    continue;
                }
                for (Tuple2<Double, Object> currentValue : values) {
                    head = updateQueue(priorityQueue, topN, currentValue, head);
                }
            }
        }

        /**
         * update queue.
         *
         * @param map queue.
         * @param topN top N.
         * @param newValue new value.
         * @param head head value.
         * @param <T> id type.
         * @return head value.
         */
        private <T> Tuple2<Double, T> updateQueue(
                PriorityQueue<Tuple2<Double, T>> map,
                int topN,
                Tuple2<Double, T> newValue,
                Tuple2<Double, T> head) {
            if (null == newValue) {
                return head;
            }
            if (map.size() < topN) {
                map.add(Tuple2.of(newValue.f0, newValue.f1));
                head = map.peek();
            } else {
                if (map.comparator().compare(head, newValue) < 0) {
                    Tuple2<Double, T> peek = map.poll();
                    peek.f0 = newValue.f0;
                    peek.f1 = newValue.f1;
                    map.add(peek);
                    head = map.peek();
                }
            }
            return head;
        }

        /**
         * prepare sample.
         *
         * @param input sample to parse.
         * @return
         */
        private Object prepareSample(Object input, KnnModelData modelData) {
            return modelData
                    .getFastDistance()
                    .prepareVectorData(Tuple2.of((DenseVector) input, null));
        }

        private ArrayList<Tuple2<Double, Object>> computeDistance(Object input, Integer index) {
            FastDistanceMatrixData data = modelData.getDictData().get(index);
            DenseMatrix res =
                    modelData.getFastDistance().calc((FastDistanceVectorData) input, data);
            ArrayList<Tuple2<Double, Object>> list = new ArrayList<>(0);
            String[] curRows = data.getIds();
            for (int i = 0; i < data.getIds().length; i++) {
                Tuple2<Double, Object> tuple = Tuple2.of(res.getData()[i], curRows[i]);
                list.add(tuple);
            }
            return list;
        }

        /**
         * get output format of knn predict result.
         *
         * @param tuple initial result from knn predictor.
         * @return output format result.
         */
        private Tuple2<Object, String> getResultFormat(Tuple2<List<Object>, List<Object>> tuple)
                throws JsonProcessingException {
            double percent = 1.0 / tuple.f0.size();
            Map<Object, Double> detail = new HashMap<>(0);

            for (Object obj : tuple.f0) {
                detail.merge(obj, percent, Double::sum);
            }

            double max = 0.0;
            Object prediction = null;

            for (Map.Entry<Object, Double> entry : detail.entrySet()) {
                if (entry.getValue() > max) {
                    max = entry.getValue();
                    prediction = entry.getKey();
                }
            }

            return Tuple2.of(prediction, ReadWriteUtils.OBJECT_MAPPER.writeValueAsString(detail));
        }

        /**
         * @param json json format result of knn prediction.
         * @param idType id type.
         * @return List format result.
         */
        private Tuple2<List<Object>, List<Object>> extractObject(String json, DataType idType)
                throws Exception {
            Map<String, String> deserializedJson;
            try {
                deserializedJson = ReadWriteUtils.OBJECT_MAPPER.readValue(json, HashMap.class);
            } catch (Exception e) {
                throw new IllegalStateException(
                        "Fail to deserialize json '" + json + "', please check the input!");
            }

            Map<String, String> lowerCaseDeserializedJson = new HashMap<>(0);

            for (Map.Entry<String, String> entry : deserializedJson.entrySet()) {
                lowerCaseDeserializedJson.put(
                        entry.getKey().trim().toLowerCase(), entry.getValue());
            }

            Map<String, List<Object>> map = new HashMap<>(2);

            String ids = lowerCaseDeserializedJson.get("id");
            String metric = lowerCaseDeserializedJson.get("metric");
            map.put("id", ReadWriteUtils.OBJECT_MAPPER.readValue(ids, List.class));
            map.put("metric", ReadWriteUtils.OBJECT_MAPPER.readValue(metric, List.class));
            return Tuple2.of(map.get("id"), map.get("metric"));
        }

        private void loadModel(List<Object> broadcastVar) throws JsonProcessingException {
            List<FastDistanceMatrixData> dictData = new ArrayList<>();
            for (Object obj : broadcastVar) {
                Row row = (Row) obj;
                if (row.getField(0) != null) {
                    dictData.add(FastDistanceMatrixData.fromString((String) row.getField(0)));
                }
            }

            modelData = new KnnModelData(dictData, new FastDistance());
        }
    }

    /** @return parameters for algorithm. */
    @Override
    public Map<Param<?>, Object> getParamMap() {
        return this.params;
    }

    @Override
    public void save(String path) throws IOException {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelData[0]).getTableEnvironment();

        String dataPath = ReadWriteUtils.getDataPath(path);
        FileSink<Row> sink =
                FileSink.forRowFormat(new Path(dataPath), new KnnModelData.ModelDataEncoder())
                        .withRollingPolicy(OnCheckpointRollingPolicy.build())
                        .withBucketAssigner(new BasePathBucketAssigner<>())
                        .build();
        tEnv.toDataStream(modelData[0]).sinkTo(sink);
        HashMap<String, String> meta = new HashMap<>(1);
        meta.put("idType", modelData[0].getResolvedSchema().getColumnDataTypes().get(1).toString());
        ReadWriteUtils.saveMetadata(this, path, meta);
    }

    public static KnnModel load(StreamExecutionEnvironment env, String path) throws IOException {
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        KnnModel retModel = ReadWriteUtils.loadStageParam(path);
        Map<String, ?> meta = ReadWriteUtils.loadMetadata(path, "");
        String idTypeStr = (String) meta.get("idType");
        DataType idType =
                LogicalTypeDataTypeConverter.toDataType(LogicalTypeParser.parse(idTypeStr));
        Source<Row, ?, ?> source =
                FileSource.forRecordStreamFormat(
                                new KnnModelData.ModelDataStreamFormat(idType),
                                ReadWriteUtils.getDataPaths(path))
                        .build();
        DataStream<Row> modelDataStream =
                env.fromSource(source, WatermarkStrategy.noWatermarks(), "data");
        retModel.modelData =
                new Table[] {
                    tEnv.fromDataStream(modelDataStream, KnnModelData.getModelSchema(idType))
                };
        return retModel;
    }
}
