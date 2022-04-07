package org.apache.flink.ml.classification.ftrl;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.ObjectArrayTypeInfo;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.classification.logisticregression.LinearModelData;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.functions.windowing.AllWindowFunction;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.api.windowing.windows.GlobalWindow;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Ftrl implements Estimator <Ftrl, FtrlModel>, FtrlParams <Ftrl> {
	private final Map <Param <?>, Object> paramMap = new HashMap <>();
	private Table initModelDataTable;

	public Ftrl() {
		ParamUtils.initializeMapWithDefaultValues(paramMap, this);
	}

	@Override
	@SuppressWarnings("unchecked")
	public FtrlModel fit(Table... inputs) {
		Preconditions.checkArgument(inputs.length == 1);

		StreamTableEnvironment tEnv =
			(StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

		DataStream <Tuple2 <Vector, Double>> points = tEnv.toDataStream(inputs[0])
			.map(new ParseSample(getFeaturesCol(), getLabelCol()));

		DataStream <LinearModelData> modelDataStream =
			LinearModelData.getModelDataStream(initModelDataTable);
		DataStream <DenseVector[]> initModelData = modelDataStream.map(new GetVectorData());
		initModelData.getTransformation().setParallelism(1);

		IterationBody body = new FtrlIterationBody(getGlobalBatchSize(), getAlpha(), getBETA(), getL1(), getL2());

		DataStream <LinearModelData> onlineModelData =
			Iterations.iterateUnboundedStreams(
					DataStreamList.of(initModelData), DataStreamList.of(points), body)
				.get(0);

		Table onlineModelDataTable = tEnv.fromDataStream(onlineModelData);
		FtrlModel model = new FtrlModel().setModelData(onlineModelDataTable);
		ReadWriteUtils.updateExistingParams(model, paramMap);
		return model;
	}

	public static class FtrlIterationBody implements IterationBody {
		private final int batchSize;
		private final double alpha;
		private final double beta;
		private final double l1;
		private final double l2;

		public FtrlIterationBody(int batchSize, double alpha, double beta, double l1, double l2) {
			this.batchSize = batchSize;
			this.alpha = alpha;
			this.beta = beta;
			this.l1 = l1;
			this.l2 = l2;
		}

		@Override
		public IterationBodyResult process(
			DataStreamList variableStreams, DataStreamList dataStreams) {
			DataStream <DenseVector[]> modelData = variableStreams.get(0);

			DataStream <Tuple2 <Vector, Double>> points = dataStreams.get(0);

			int parallelism = points.getParallelism();
			DataStream <DenseVector[]> newModelData =
				points.countWindowAll(batchSize)
					.apply(new GlobalBatchCreator())
					.flatMap(new GlobalBatchSplitter(parallelism))
					.rebalance()
					.connect(modelData.broadcast())
					.transform(
						"ModelDataLocalUpdater",
						TypeInformation.of(DenseVector[].class),
						new FtrlLocalUpdater(alpha, beta, l1, l2))
					.setParallelism(parallelism)
					.countWindowAll(parallelism)
					.reduce(new FtrlGlobalReducer())
					.map((MapFunction <DenseVector[], DenseVector[]>) value -> new DenseVector[]{value[0]}).setParallelism(1);

			return new IterationBodyResult(DataStreamList.of(newModelData), DataStreamList.of(modelData.map(
				new MapFunction <DenseVector[], LinearModelData>() {
					long iter = 0L;
					@Override
					public LinearModelData map(DenseVector[] value) {
						return new LinearModelData(value[0], iter++);
					}
				})));
		}
	}

	public static class GetVectorData implements MapFunction<LinearModelData, DenseVector[]> {
		@Override
		public DenseVector[] map(LinearModelData value) throws Exception {
			return new DenseVector[] {
				value.coefficient};
		}
	}

	/**
	 * Operator that collects a LogisticRegressionModelData from each upstream subtask, and outputs the weight average
	 * of collected model data.
	 */
	public static class FtrlGlobalReducer implements ReduceFunction <DenseVector[]> {
		@Override
		public DenseVector[] reduce(DenseVector[] modelData, DenseVector[] newModelData) {
			for (int i = 0; i < newModelData[0].size(); ++i) {
				if ((modelData[1].values[i] + newModelData[1].values[i]) > 0.0) {
					newModelData[0].values[i] = (modelData[0].values[i] * modelData[1].values[i] +
						newModelData[0].values[i] * newModelData[1].values[i]) / (modelData[1].values[i] +
						newModelData[1].values[i]);
				}
				newModelData[1].values[i] = modelData[1].values[i] + newModelData[1].values[i];
			}
			return newModelData;
		}
	}

	public static class FtrlLocalUpdater extends AbstractStreamOperator <DenseVector[]>
		implements TwoInputStreamOperator <Tuple2 <Vector, Double>[], DenseVector[], DenseVector[]> {
		private ListState <Tuple2 <Vector, Double>[]> localBatchDataState;
		private ListState <DenseVector[]> modelDataState;
		private double[] N;
		private double[] Z;
		private final double alpha;
		private final double beta;
		private final double l1;
		private final double l2;
		private DenseVector weights;
		public FtrlLocalUpdater(double alpha, double beta, double l1, double l2) {
			this.alpha = alpha;
			this.beta = beta;
			this.l1 = l1;
			this.l2 = l2;
		}

		@Override
		public void initializeState(StateInitializationContext context) throws Exception {
			super.initializeState(context);

			TypeInformation <Tuple2 <Vector, Double>[]> type =
				ObjectArrayTypeInfo.getInfoFor(DenseVectorTypeInfo.INSTANCE);
			localBatchDataState =
				context.getOperatorStateStore()
					.getListState(new ListStateDescriptor <>("localBatch", type));
			modelDataState =
				context.getOperatorStateStore()
					.getListState(
						new ListStateDescriptor <>("modelData", DenseVector[].class));
		}

		@Override
		public void processElement1(StreamRecord <Tuple2 <Vector, Double>[]> pointsRecord) throws Exception {
			localBatchDataState.add(pointsRecord.getValue());
			alignAndComputeModelData();
		}

		@Override
		public void processElement2(StreamRecord <DenseVector[]> modelDataRecord)
			throws Exception {
			modelDataState.add(modelDataRecord.getValue());
			alignAndComputeModelData();
		}

		private void alignAndComputeModelData() throws Exception {
			if (!modelDataState.get().iterator().hasNext()
				|| !localBatchDataState.get().iterator().hasNext()) {
				return;
			}

			DenseVector[] modelData =
				OperatorStateUtils.getUniqueElement(modelDataState, "modelData").get();
			modelDataState.clear();

			List <Tuple2 <Vector, Double>[]> pointsList =
				IteratorUtils.toList(localBatchDataState.get().iterator());
			Tuple2 <Vector, Double>[] points = pointsList.remove(0);
			localBatchDataState.update(pointsList);

			for (Tuple2 <Vector, Double> point : points) {
				if (N == null) {
					N = new double[point.f0.size()];
					Z = new double[N.length];
					weights = new DenseVector(N.length);
				}

				double p = 0.0;
				Arrays.fill(weights.values, 0.0);
				if (point.f0 instanceof DenseVector) {
					DenseVector denseVector = (DenseVector) point.f0;
					for (int i = 0; i < denseVector.size(); ++i) {
						if (Math.abs(Z[i]) <= l1) {
							modelData[0].values[i] = 0.0;
						} else {
							modelData[0].values[i] = ((Z[i] < 0 ? -1 : 1) * l1 - Z[i]) / ((beta + Math.sqrt(
								N[i])) / alpha + l2);
						}
						p += modelData[0].values[i] * denseVector.values[i];
					}
					p = 1 / (1 + Math.exp(-p));
					for (int i = 0; i < denseVector.size(); ++i) {
						double g = (p - point.f1) * denseVector.values[i];
						double sigma = (Math.sqrt(N[i] + g * g) - Math.sqrt(N[i])) / alpha;
						Z[i] += g - sigma * modelData[0].values[i];
						N[i] += g * g;
						weights.values[i] += 1.0;
					}
				} else {
					SparseVector sparseVector = (SparseVector) point.f0;
					for (int i = 0; i < sparseVector.indices.length; ++i) {
						int idx = sparseVector.indices[i];
						if (Math.abs(Z[idx]) <= l1) {
							modelData[0].values[idx] = 0.0;
						} else {
							modelData[0].values[idx] = ((Z[idx] < 0 ? -1 : 1) * l1 - Z[idx]) / ((beta + Math.sqrt(
								N[idx])) / alpha + l2);
						}
						p += modelData[0].values[idx] * sparseVector.values[i];
					}
					p = 1 / (1 + Math.exp(-p));
					for (int i = 0; i < sparseVector.indices.length; ++i) {
						int idx = sparseVector.indices[i];
						double g = (p - point.f1) * sparseVector.values[i];
						double sigma = (Math.sqrt(N[idx] + g * g) - Math.sqrt(N[idx])) / alpha;
						Z[idx] += g - sigma * modelData[0].values[idx];
						N[idx] += g * g;
						weights.values[idx] += 1.0;
					}
				}
			}
			output.collect(new StreamRecord <>(new DenseVector[]{modelData[0], weights}));
		}
	}

	public static class ParseSample
		extends RichMapFunction <Row, Tuple2 <Vector, Double>> {
		private static final long serialVersionUID = 3738888745125082777L;

		private final String featureCol;
		private final String labelCol;

		public ParseSample(String featureCol, String labelCol) {
			this.featureCol = featureCol;
			this.labelCol = labelCol;
		}

		@Override
		public Tuple2 <Vector, Double> map(Row row) {
			return Tuple2.of((Vector) row.getField(featureCol), (Double) row.getField(labelCol));
		}
	}

	@Override
	public void save(String path) {

	}

	@Override
	public Map <Param <?>, Object> getParamMap() {
		return paramMap;
	}

	/**
	 * An operator that splits a global batch into evenly-sized local batches, and distributes them to downstream
	 * operator.
	 */
	private static class GlobalBatchSplitter
		implements FlatMapFunction <Tuple2 <Vector, Double>[], Tuple2 <Vector, Double>[]> {
		private final int downStreamParallelism;

		private GlobalBatchSplitter(int downStreamParallelism) {
			this.downStreamParallelism = downStreamParallelism;
		}

		@Override
		public void flatMap(Tuple2 <Vector, Double>[] values,
							Collector <Tuple2 <Vector, Double>[]> collector) {
			int div = values.length / downStreamParallelism;
			int mod = values.length % downStreamParallelism;

			int offset = 0;
			int i = 0;

			int size = div + 1;
			for (; i < mod; i++) {
				collector.collect(Arrays.copyOfRange(values, offset, offset + size));
				offset += size;
			}

			size = div;
			for (; i < downStreamParallelism; i++) {
				collector.collect(Arrays.copyOfRange(values, offset, offset + size));
				offset += size;
			}
		}
	}

	private static class GlobalBatchCreator
		implements AllWindowFunction <Tuple2 <Vector, Double>, Tuple2 <Vector, Double>[], GlobalWindow> {
		@Override
		public void apply(
			GlobalWindow timeWindow,
			Iterable <Tuple2 <Vector, Double>> iterable,
			Collector <Tuple2 <Vector, Double>[]> collector) {
			List <Tuple2 <Vector, Double>> points = IteratorUtils.toList(iterable.iterator());
			collector.collect(points.toArray(new Tuple2[0]));
		}
	}

	/**
	 * Sets the initial model data of the online training process with the provided model data table.
	 */
	public Ftrl setInitialModelData(Table initModelDataTable) {
		this.initModelDataTable = initModelDataTable;
		return this;
	}

}
