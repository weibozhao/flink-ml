package org.apache.flink.ml.algo.batch.knn;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.common.BatchOperator;
import org.apache.flink.ml.common.MapPartitionFunctionWrapper;
import org.apache.flink.ml.common.linalg.DenseVector;
import org.apache.flink.ml.common.linalg.VectorUtil;

import org.apache.flink.ml.algo.batch.knn.distance.BaseFastDistance;
import org.apache.flink.ml.algo.batch.knn.distance.BaseFastDistanceData;
import org.apache.flink.ml.algo.batch.knn.distance.FastDistanceMatrixData;
import org.apache.flink.ml.algo.batch.knn.distance.FastDistanceSparseData;
import org.apache.flink.ml.algo.batch.knn.distance.FastDistanceVectorData;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.StringParam;
import org.apache.flink.ml.params.knn.HasKnnDistanceType;
import org.apache.flink.ml.params.knn.KnnTrainParams;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.table.types.DataType;
import org.apache.flink.table.types.utils.TypeConversions;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.flink.ml.algo.batch.knn.distance.BaseFastDistanceData.pGson;

/**
 * KNN is to classify unlabeled observations by assigning them to the class of the most similar labeled examples. Note
 * that though there is no ``training process`` in KNN, we create a ``fake one`` to use in pipeline model. In this
 * operator, we do some preparation to speed up the inference process.
 */
public final class KnnTrainBatchOp extends BatchOperator <KnnTrainBatchOp>
	implements KnnTrainParams <KnnTrainBatchOp> {
	private static final long serialVersionUID = -3118065094037473283L;
	private static final int ROW_SIZE = 2;
	private static final int FASTDISTANCE_TYPE_INDEX = 0;
	private static final int DATA_INDEX = 1;

	public static Param <String> ID_TYPE =
		new StringParam("idType", "id type", null);

	/**
	 * constructor.
	 */
	public KnnTrainBatchOp() {
		this(new HashMap <>());
	}

	/**
	 * constructor.
	 *
	 * @param params parameters for algorithm.
	 */
	public KnnTrainBatchOp(Map <Param <?>, Object> params) {
		super(params);
	}

	@Override
	public void save(String path) throws IOException {}

	/**
	 * train knn model.
	 *
	 * @param inputs a list of tables.
	 * @return a list of tables, which containing knn model table.
	 */
	@Override
	public Table[] transform(Table... inputs) {
		Preconditions.checkArgument(inputs.length == 1);
		ResolvedSchema schema = inputs[0].getResolvedSchema();
		String[] colNames = schema.getColumnNames().toArray(new String[0]);
		StreamTableEnvironment tEnv =
			(StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
		DataStream <Row> input =
				tEnv.toDataStream(inputs[0]);
		final int[] featureIndices = KnnUtils.findColIndices(colNames, getFeatureCols());
		String labelCol = getLabelCol();
		final int labelIdx = KnnUtils.findColIndex(colNames, labelCol);
		final int vecIdx =
			getVectorCol() != null
				? KnnUtils.findColIndex(
				inputs[0]
					.getResolvedSchema()
					.getColumnNames()
					.toArray(new String[0]),
				getVectorCol())
				: -1;

		DataStream <Row> trainData =
			input.map(
				(MapFunction <Row, Row>)
					value -> {
						Object label = value.getField(labelIdx);
						if (vecIdx == -1) {
							DenseVector vec = new DenseVector(featureIndices.length);
							for (int i = 0; i < featureIndices.length; ++i) {
								vec.set(
									i,
									((Number) value.getField(featureIndices[i]))
										.doubleValue());
							}
							return Row.of(label, vec);
						} else {
							DenseVector vec =
								(DenseVector)
									VectorUtil.parse(
										value.getField(vecIdx).toString());
							return Row.of(label, vec);
						}
					});

		DataType idType =
		        schema.getColumnDataTypes().get(KnnUtils.findColIndex(colNames, labelCol));
		DataStream <Row> model = buildModel(trainData, getParamMap(), idType);
		return new Table[] {tEnv.fromDataStream(model)};
	}

	/**
	 * build knn model.
	 *
	 * @param dataStream input data.
	 * @param params     input parameters.
	 * @return stream format model.
	 */
	public static DataStream <Row> buildModel(
		DataStream <Row> dataStream, final Map <Param <?>, Object> params, final DataType idType) {
		BaseFastDistance fastDistance = ((DistanceType) params.get(HasKnnDistanceType.DISTANCE_TYPE)).getFastDistance();
		DataStream <Row> index =
			dataStream.transform(
				"build index",
				dataStream.getType(),
				new MapPartitionFunctionWrapper <>(
					"build index",
					dataStream.getType(),
					new RichMapPartitionFunction <Row, Row>() {

						@Override
						public void mapPartition(
							Iterable <Row> values, Collector <Row> out)
							throws Exception {
							List <BaseFastDistanceData> list =
								fastDistance.prepareMatrixData(values, 1, 0);
							for (BaseFastDistanceData fastDistanceData : list) {
								Row row = new Row(ROW_SIZE);
								if (fastDistanceData
									instanceof FastDistanceMatrixData) {
									row.setField(FASTDISTANCE_TYPE_INDEX, 1L);
									FastDistanceMatrixData data =
										(FastDistanceMatrixData) fastDistanceData;
									row.setField(DATA_INDEX, data.toString());
								} else if (fastDistanceData
									instanceof FastDistanceVectorData) {
									row.setField(FASTDISTANCE_TYPE_INDEX, 2L);
									FastDistanceVectorData data =
										(FastDistanceVectorData) fastDistanceData;
									row.setField(DATA_INDEX, data.toString());
								} else if (fastDistanceData
									instanceof FastDistanceSparseData) {
									row.setField(FASTDISTANCE_TYPE_INDEX, 3L);
									FastDistanceSparseData data =
										(FastDistanceSparseData) fastDistanceData;
									row.setField(DATA_INDEX, data.toString());
								} else {
									throw new RuntimeException(
										fastDistanceData.getClass().getName()
											+ "is not supported!");
								}
								out.collect(row);

							}
						}
					}));

		return index
			.transform(
				"buildKnnModel",
				new RowTypeInfo(
					new TypeInformation[] {
						Types.LONG, Types.STRING, Types.STRING,
						TypeInformation.of(idType.getLogicalType().getDefaultConversion())
						//TypeConversions.fromDataTypeToLegacyInfo(idType)
					},
					new String[] {"VECTOR_TYPE", "DATA", "META", "VECTORMODELDATACONVERTER"}),
				new MapPartitionFunctionWrapper <>(
					"buildKnnModel",
					index.getType(),
					new RichMapPartitionFunction <Row, Row>() {
						private static final long serialVersionUID = 661383020005730224L;

						@Override
						public void open(Configuration parameters) throws Exception {
							super.open(parameters);
						}

						@Override
						public void mapPartition(Iterable <Row> values, Collector <Row> out)
							throws Exception {
							Map <String, Object> meta = new HashMap <>(params.size());
							if (getRuntimeContext().getIndexOfThisSubtask() == 0) {
								for (Param <?> key : params.keySet()) {
									if (params.get(key) != null) {
										meta.put(key.name, pGson.toJson(params.get(key)));
									}
								}
								meta.put("idType", idType.toString());
							}

							if (meta.size() > 0) {
								Row row = new Row(ROW_SIZE + 2);
								row.setField(ROW_SIZE, pGson.toJson(meta));
								out.collect(row);
							}

							for (Row r : values) {
								out.collect(KnnUtils.merge(r, new Row(2)));
							}
						}
					}));
	}
}
