package org.apache.flink.ml.classification.ftrl;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.metrics.Gauge;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.classification.logisticregression.LinearModelData;

import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class FtrlModel implements Model <FtrlModel>, FtrlModelParams <FtrlModel> {
	private final Map <Param <?>, Object> paramMap = new HashMap <>();
	private Table modelDataTable;

	public FtrlModel() {
		ParamUtils.initializeMapWithDefaultValues(paramMap, this);
	}

	@Override
	public Table[] transform(Table... inputs) {
		Preconditions.checkArgument(inputs.length == 1);

		StreamTableEnvironment tEnv =
			(StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

		RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
		RowTypeInfo outputTypeInfo =
			new RowTypeInfo(
				ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), Types.DOUBLE, Types.LONG),
				ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getPredictionCol(), "modelVersion"));

		DataStream <Row> predictionResult =
			tEnv.toDataStream(inputs[0])
				.connect(LinearModelData.getModelDataStream(modelDataTable).broadcast())
				.transform(
					"PredictLabelOperator",
					outputTypeInfo,
					new PredictLabelOperator(
						inputTypeInfo,
						getFeaturesCol()));

		return new Table[] {tEnv.fromDataStream(predictionResult)};
	}

	/** A utility operator used for prediction. */
	private static class PredictLabelOperator extends AbstractStreamOperator <Row>
		implements TwoInputStreamOperator <Row, LinearModelData, Row> {
		private final RowTypeInfo inputTypeInfo;

		private final String featuresCol;
		private ListState <Row> bufferedPointsState;
		private DenseVector coefficient;
		private long modelDataVersion = 0;

		public PredictLabelOperator(
			RowTypeInfo inputTypeInfo,
			String featuresCol) {
			this.inputTypeInfo = inputTypeInfo;
			this.featuresCol = featuresCol;
		}

		@Override
		public void initializeState(StateInitializationContext context) throws Exception {
			super.initializeState(context);

			bufferedPointsState =
				context.getOperatorStateStore()
					.getListState(
						new ListStateDescriptor <>("bufferedPoints", inputTypeInfo));
		}

		@Override
		public void open() throws Exception {
			super.open();

			getRuntimeContext()
				.getMetricGroup()
				.gauge(
					"MODEL_DATA_VERSION_GAUGE_KEY",
					(Gauge <String>) () -> Long.toString(modelDataVersion));
		}

		@Override
		public void processElement1(StreamRecord <Row> streamRecord) throws Exception {
			Row dataPoint = streamRecord.getValue();
			// todo : predict data
			Vector features = (Vector) dataPoint.getField(featuresCol);
			Tuple2 <Double, DenseVector> predictionResult = predictRaw(features, coefficient);
			output.collect(new StreamRecord<>(Row.join(dataPoint, Row.of(predictionResult.f0, modelDataVersion))));
		}

		@Override
		public void processElement2(StreamRecord<LinearModelData> streamRecord) throws Exception {
			LinearModelData modelData = streamRecord.getValue();

			coefficient = modelData.coefficient;
			modelDataVersion = modelData.modelVersion;
			//System.out.println("update model...");
			//todo : receive model data.
			// Preconditions.checkArgument(modelData.centroids.length <= k);
			//centroids = modelData.centroids;
			//modelDataVersion++;
			//for (Row dataPoint : bufferedPointsState.get()) {
			//	processElement1(new StreamRecord<>(dataPoint));
			//}
			//bufferedPointsState.clear();
		}
	}

	/**
	 * The main logic that predicts one input record.
	 *
	 * @param feature The input feature.
	 * @param coefficient The model parameters.
	 * @return The prediction label and the raw probabilities.
	 */
	public static Tuple2<Double, DenseVector> predictRaw(
		Vector feature, DenseVector coefficient) throws InterruptedException {
		while (coefficient == null) {
			Thread.sleep(100);
		}
		double dotValue = 0.0;
		if (feature instanceof SparseVector) {
			SparseVector svec = (SparseVector) feature;
			for (int i = 0; i < svec.indices.length; ++i) {
				dotValue += svec.values[i] * coefficient.values[svec.indices[i]];
			}
		} else {
			dotValue = BLAS.dot((DenseVector) feature, coefficient);
		}
		double prob = 1 - 1.0 / (1.0 + Math.exp(dotValue));
		return new Tuple2<>(dotValue >= 0 ? 1. : 0., Vectors.dense(1 - prob, prob));
	}

	@Override
	public void save(String path) throws IOException {

	}

	@Override
	public Map <Param <?>, Object> getParamMap() {
		return paramMap;
	}

	@Override
	public FtrlModel setModelData(Table... inputs) {
		modelDataTable = inputs[0];
		return this;
	}

	@Override
	public Table[] getModelData() {
		return new Table[] {modelDataTable};
	}
}
