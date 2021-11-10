package org.apache.flink.ml.algo.batch.knn;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.algo.batch.knn.distance.BaseFastDistance;
import org.apache.flink.ml.algo.batch.knn.distance.BaseFastDistanceData;
import org.apache.flink.ml.algo.batch.knn.distance.FastDistanceMatrixData;
import org.apache.flink.ml.algo.batch.knn.distance.FastDistanceSparseData;
import org.apache.flink.ml.algo.batch.knn.distance.FastDistanceVectorData;
import org.apache.flink.ml.common.BatchOperator;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.linalg.DenseVector;
import org.apache.flink.ml.common.linalg.Vector;
import org.apache.flink.ml.common.linalg.VectorUtil;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.params.knn.HasKnnDistanceType.DistanceType;
import org.apache.flink.ml.params.knn.KnnPredictParams;
import org.apache.flink.ml.params.knn.KnnTrainParams;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractUdfStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.table.types.DataType;
import org.apache.flink.table.types.logical.utils.LogicalTypeParser;
import org.apache.flink.table.types.utils.LogicalTypeDataTypeConverter;
import org.apache.flink.types.Row;

import org.apache.flink.shaded.curator4.com.google.common.base.Preconditions;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.core.type.TypeReference;

import org.apache.commons.lang3.ArrayUtils;
import sun.reflect.generics.reflectiveObjects.ParameterizedTypeImpl;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.apache.flink.ml.algo.batch.knn.distance.BaseFastDistanceData.pGson;

/**
 * batch map batch operator. you can inherit this class to develop model-based prediction batch
 * operator.
 */
public class KnnPredictBatchOp extends BatchOperator<KnnPredictBatchOp>
        implements KnnPredictParams<KnnPredictBatchOp> {

    private static final long serialVersionUID = -3118065094037473283L;

    public static final String BROADCAST_STR = "broadcastModelKey";
    private static final int FASTDISTANCE_TYPE_INDEX = 0;
    private static final int DATA_INDEX = 1;

    public KnnPredictBatchOp(Map<Param<?>, Object> params) {
        super(params);
    }

    /**
     * @param inputs a list of tables
     * @return result tables.
     */
    @Override
    public Table[] transform(Table... inputs) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[1]).getTableEnvironment();
        DataStream<Row> input = tEnv.toDataStream(inputs[0]);
        DataStream<Row> model = tEnv.toDataStream(inputs[1]);

        Map<String, DataStream<?>> broadcastMap = new HashMap<>(1);
        broadcastMap.put(BROADCAST_STR, model);
        ResolvedSchema modelSchema = inputs[1].getResolvedSchema();
        DataType idType =
                modelSchema.getColumnDataTypes().get(modelSchema.getColumnNames().size() - 1);

        ResolvedSchema outputSchema =
                getOutputSchema(inputs[0].getResolvedSchema(), getParamMap(), idType);

        DataType[] dataTypes = outputSchema.getColumnDataTypes().toArray(new DataType[0]);
        TypeInformation<?>[] typeInformations = new TypeInformation[dataTypes.length];

        for (int i = 0; i < dataTypes.length; ++i) {
            typeInformations[i] = TypeInformation.of(dataTypes[i].getLogicalType().getClass());
        }

        Function<List<DataStream<?>>, DataStream<Row>> function =
                dataStreams -> {
                    DataStream stream = dataStreams.get(0);
                    return stream.transform(
                            "mapFunc",
                            new RowTypeInfo(
                                    typeInformations,
                                    outputSchema.getColumnNames().toArray(new String[0])),
                            new PredictOperator(
                                    new KnnRichFunction(
                                            getParamMap(), inputs[0].getResolvedSchema())));
                };

        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(input), broadcastMap, function);
        return new Table[] {
            tEnv.fromDataStream(output, KnnUtils.resolvedSchema2Schema(outputSchema))
        };
    }

    private static class KnnRichFunction extends RichMapFunction<Row, Row> {
        private boolean firstEle = true;
        private final boolean isPredDetail;

        private String[] reservedCols;
        private String[] selectedCols;
        private String vectorCol;
        private DataType idType;
        private transient KnnModelData modelData;
        private final Integer topN;
        private Map<String, Object> meta;

        public KnnRichFunction(Map<Param<?>, Object> params, ResolvedSchema dataSchema) {
            reservedCols = (String[]) params.get(KnnPredictParams.RESERVED_COLS);
            reservedCols =
                    (reservedCols == null)
                            ? dataSchema.getColumnNames().toArray(new String[0])
                            : reservedCols;
            isPredDetail = params.containsKey(KnnPredictParams.PREDICTION_DETAIL_COL);
            this.topN = (Integer) params.get(KnnPredictParams.K);
        }

        @Override
        public Row map(Row row) throws Exception {
            if (firstEle) {
                loadModel(getRuntimeContext().getBroadcastVariable(BROADCAST_STR));
                firstEle = false;
            }
            Vector vector;
            if (null != selectedCols) {
                vector = new DenseVector(selectedCols.length);
                for (int i = 0; i < selectedCols.length; i++) {
                    Preconditions.checkNotNull(
                            row.getField(selectedCols[i]), "There is NULL in featureCols!");
                    vector.set(i, ((Number) row.getField(selectedCols[i])).doubleValue());
                }
            } else {
                vector = VectorUtil.parse(row.getField(vectorCol).toString());
            }
            String s = modelData.findNeighbor(vector, topN, null).toLowerCase();

            Row ret = new Row(reservedCols.length + (isPredDetail ? 2 : 1));
            for (int i = 0; i < reservedCols.length; ++i) {
                ret.setField(i, row.getField(reservedCols[i]));
            }

            Tuple2<Object, String> tuple2 = getResultFormat(extractObject(s, idType));
            ret.setField(reservedCols.length, tuple2.f0);
            if (isPredDetail) {
                ret.setField(reservedCols.length + 1, tuple2.f1);
            }
            return ret;
        }

        /**
         * get output format of knn predict result.
         *
         * @param tuple initial result from knn predictor.
         * @return output format result.
         */
        private Tuple2<Object, String> getResultFormat(Tuple2<List<Object>, List<Object>> tuple) {
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

            return Tuple2.of(prediction, pGson.toJson(detail));
        }

        /**
         * @param json json format result of knn prediction.
         * @param idType id type.
         * @return List format result.
         */
        private static Tuple2<List<Object>, List<Object>> extractObject(
                String json, DataType idType) {
            Map<String, String> deserializedJson;
            try {
                deserializedJson =
                        pGson.fromJson(json, new TypeReference<Map<String, String>>() {}.getType());
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

            Type type = idType.getLogicalType().getDefaultConversion();
            String ids = lowerCaseDeserializedJson.get("id");
            String metric = lowerCaseDeserializedJson.get("metric");
            if (ids == null) {
                map.put("id", null);
            } else {
                map.put(
                        "id",
                        pGson.fromJson(
                                ids,
                                ParameterizedTypeImpl.make(List.class, new Type[] {type}, null)));
            }

            if (ids == null) {
                map.put("metric", null);
            } else {
                map.put(
                        "metric",
                        pGson.fromJson(
                                metric,
                                ParameterizedTypeImpl.make(
                                        List.class, new Type[] {Double.class}, null)));
            }
            return Tuple2.of(map.get("id"), map.get("metric"));
        }

        public void loadModel(List<Object> broadcastVar) {
            List<BaseFastDistanceData> dictData = new ArrayList<>();
            for (Object obj : broadcastVar) {
                Row row = (Row) obj;
                if (row.getField(row.getArity() - 2) != null) {
                    meta = pGson.fromJson((String) row.getField(row.getArity() - 2), HashMap.class);
                }
            }
            for (Object obj : broadcastVar) {
                Row row = (Row) obj;
                if (row.getField(FASTDISTANCE_TYPE_INDEX) != null) {
                    long type = (long) row.getField(FASTDISTANCE_TYPE_INDEX);
                    if (type == 1L) {
                        dictData.add(
                                FastDistanceMatrixData.fromString(
                                        (String) row.getField(DATA_INDEX)));
                    } else if (type == 2L) {
                        dictData.add(
                                FastDistanceVectorData.fromString(
                                        (String) row.getField(DATA_INDEX)));
                    } else if (type == 3L) {
                        dictData.add(
                                FastDistanceSparseData.fromString(
                                        (String) row.getField(DATA_INDEX)));
                    }
                }
            }
            if (meta.containsKey(KnnTrainParams.FEATURE_COLS.name)) {
                selectedCols =
                        pGson.fromJson(
                                (String) meta.get(KnnTrainParams.FEATURE_COLS.name),
                                String[].class);
            } else {
                vectorCol =
                        pGson.fromJson(
                                (String) meta.get(KnnTrainParams.VECTOR_COL.name), String.class);
            }

            BaseFastDistance distance =
                    pGson.fromJson(
                                    (String) meta.get(KnnTrainParams.DISTANCE_TYPE.name),
                                    DistanceType.class)
                            .getFastDistance();
            modelData = new KnnModelData(dictData, distance);
            idType =
                    LogicalTypeDataTypeConverter.toDataType(
                            LogicalTypeParser.parse((String) this.meta.get("idType")));
            modelData.setIdType(idType);
        }
    }

    /**
     * this operator use mapper to load the model data and do the prediction. if you want to write a
     * prediction operator, you need implement a special mapper for this operator.
     */
    private static class PredictOperator
            extends AbstractUdfStreamOperator<Row, RichMapFunction<Row, Row>>
            implements OneInputStreamOperator<Row, Row> {

        public PredictOperator(RichMapFunction<Row, Row> userFunction) {
            super(userFunction);
        }

        @Override
        public void processElement(StreamRecord<Row> streamRecord) throws Exception {
            Row value = streamRecord.getValue();
            output.collect(new StreamRecord<>(userFunction.map(value)));
        }
    }

    public ResolvedSchema getOutputSchema(
            ResolvedSchema dataSchema, Map<Param<?>, Object> params, DataType idType) {
        String[] reservedCols = (String[]) params.get(KnnPredictParams.RESERVED_COLS);
        reservedCols =
                (reservedCols == null)
                        ? dataSchema.getColumnNames().toArray(new String[0])
                        : reservedCols;
        DataType[] reservedTypes = KnnUtils.findColTypes(dataSchema, reservedCols);
        boolean isPredDetail = params.containsKey(KnnPredictParams.PREDICTION_DETAIL_COL);
        String[] resultCols =
                isPredDetail
                        ? new String[] {
                            (String) params.get(KnnPredictParams.PREDICTION_COL),
                            (String) params.get(KnnPredictParams.PREDICTION_DETAIL_COL)
                        }
                        : new String[] {(String) params.get(KnnPredictParams.PREDICTION_COL)};
        DataType[] resultTypes =
                isPredDetail
                        ? new DataType[] {idType, DataTypes.STRING()}
                        : new DataType[] {idType};
        return ResolvedSchema.physical(
                ArrayUtils.addAll(reservedCols, resultCols),
                ArrayUtils.addAll(reservedTypes, resultTypes));
    }

    @Override
    public void save(String path) {}
}
