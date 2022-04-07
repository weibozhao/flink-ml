package org.apache.flink.ml.classification.ftrl;

import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.Partitioner;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeHint;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.tuple.Tuple6;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.classification.logisticregression.LinearModelData;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.IterativeStream;
import org.apache.flink.streaming.api.functions.co.RichCoFlatMapFunction;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FtrlSplitVec implements Estimator <FtrlSplitVec, FtrlModel>, FtrlParams <FtrlSplitVec> {
	private final Map <Param <?>, Object> paramMap = new HashMap <>();

	public FtrlSplitVec() {
		ParamUtils.initializeMapWithDefaultValues(paramMap, this);
	}

	@Override
	@SuppressWarnings("unchecked")
	public FtrlModel fit(Table... inputs) {
		Preconditions.checkArgument(inputs.length == 2);

		StreamTableEnvironment tEnv =
			(StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

		DataStream <Row> model = tEnv.toDataStream(inputs[0]);

		DataStream <Tuple3 <Long, Vector, Double>> trainData =
			tEnv.toDataStream(inputs[1])
				.map(new ParseSample(getFeaturesCol(), getLabelCol()));

		int featureSize = getVectorSize();
		int parallelism = trainData.getParallelism();
		final int[] splitInfo = getSplitInfo(featureSize, parallelism);

		IterativeStream.ConnectedIterativeStreams <Tuple3 <Long, Vector, Double>,
			Tuple2 <Long, Object>>
			iteration = trainData.iterate(Long.MAX_VALUE)
			.withFeedbackType(TypeInformation
				.of(new TypeHint <Tuple2 <Long, Object>>() {}));

		DataStream iterativeBody = iteration.flatMap(new AppendWx())
			.flatMap(new SplitVector(splitInfo, featureSize))
			.partitionCustom(new SubVectorPartitioner(),
				(KeySelector <Tuple6 <Long, Integer, Integer, Vector, Double, double[]>, Integer>) value -> value.f1);

		iterativeBody =
			BroadcastUtils.withBroadcastStream(
				Collections.singletonList(iterativeBody),
				Collections.singletonMap("model", model),
				inputList -> {
					DataStream input = inputList.get(0);
					return input.flatMap(
						new CalcTask(splitInfo, new FtrlLearningKernel(), featureSize),
						TypeInformation.of(new TypeHint<Tuple2<Long, Object>>(){}));
				});

		iterativeBody = iterativeBody.keyBy((KeySelector <Tuple2 <Long, Object>, Long>) value -> value.f0)
			.flatMap(new ReduceTask(parallelism, splitInfo, new FtrlLearningKernel()))
			.partitionCustom(new WxPartitioner(), (KeySelector <Tuple2 <Long, Object>, Long>) value -> value.f0);

		DataStream <Tuple2 <Long, Object>>
			result = iterativeBody.filter(
			new FilterFunction <Tuple2 <Long, Object>>() {
				private static final long serialVersionUID = -5436758453355074895L;

				@Override
				public boolean filter(Tuple2 <Long, Object> t2) {
					// if t2.f0 >= 0 then feedback
					return (t2.f0 >= 0);
				}
			});

		DataStream <Tuple2 <Long, Object>> output = iterativeBody.filter(
			new FilterFunction <Tuple2 <Long, Object>>() {
				private static final long serialVersionUID = 4204787383191799107L;

				@Override
				public boolean filter(Tuple2 <Long, Object> t2) {
					/* if t2.f0 small than 0, then output */
					return t2.f0 < 0;
				}
			});

		iteration.closeWith(result);

		DataStream <DenseVector> models = output.map(
			(MapFunction <Tuple2 <Long, Object>, DenseVector>) value -> new DenseVector(
				(double[]) value.f1));

		//ReadWriteUtils.updateExistingParams(modelStream, paramMap);

		return new FtrlModel().setModelData(tEnv.fromDataStream(models));
	}

	public static class ParseSample
		extends RichMapFunction <Row, Tuple3 <Long, Vector, Double>> {
		private static final long serialVersionUID = 3738888745125082777L;
		private long counter;
		private int parallelism;
		private final String featureCol;
		private final String labelCol;

		public ParseSample(String featureCol, String labelCol) {
			this.featureCol = featureCol;
			this.labelCol = labelCol;
		}

		@Override
		public void open(Configuration parameters) {
			this.parallelism = getRuntimeContext().getNumberOfParallelSubtasks();
			counter = getRuntimeContext().getIndexOfThisSubtask();
		}

		@Override
		public Tuple3 <Long, Vector, Double> map(Row row) {
			counter += parallelism;
			return Tuple3.of(counter, (Vector) row.getField(featureCol), (Double) row.getField(labelCol));
		}
	}

	private static int[] getSplitInfo(int coefSize, int parallelism) {
		int subSize = coefSize / parallelism;
		int[] poses = new int[parallelism + 1];
		int offset = coefSize % parallelism;
		for (int i = 0; i < offset; ++i) {
			poses[i + 1] = poses[i] + subSize + 1;
		}
		for (int i = offset; i < parallelism; ++i) {
			poses[i + 1] = poses[i] + subSize;
		}
		return poses;
	}

	public static class AppendWx extends RichCoFlatMapFunction <Tuple3 <Long, Vector, Double>,
		Tuple2 <Long, Object>,
		Tuple4 <Long, Vector, Double, double[]>> {
		private static final long serialVersionUID = 7338858137860097282L;
		Map <Long, Tuple3 <Vector, Double, Long>> sampleBuffer = new HashMap <>();

		@Override
		public void flatMap1(Tuple3 <Long, Vector, Double> value,
							 Collector <Tuple4 <Long, Vector, Double, double[]>> out) {
			sampleBuffer.put(value.f0, Tuple3.of(value.f1, value.f2, System.currentTimeMillis()));
			out.collect(Tuple4.of(value.f0, value.f1, value.f2, new double[] {0.0}));
		}

		@Override
		public void flatMap2(Tuple2 <Long, Object> value, Collector <Tuple4 <Long, Vector, Double, double[]>> out) {
			Tuple3 <Vector, Double, Long> sample = sampleBuffer.get(value.f0);
			out.collect(Tuple4.of(-(value.f0 + 1), sample.f0, sample.f1, (double[]) value.f1));
		}
	}

	public static class SplitVector
		extends RichFlatMapFunction <Tuple4 <Long, Vector, Double, double[]>,
		Tuple6 <Long, Integer, Integer, Vector, Double, double[]>> {
		private static final long serialVersionUID = -8716205207637225677L;
		private int coefSize;
		private final int vectorSize;
		private int parallelism;
		private final int[] splitInfo;
		private final int[] nnz;

		public SplitVector(int[] splitInfo, int vectorSize) {
			this.vectorSize = vectorSize;

			this.splitInfo = splitInfo;
			this.nnz = new int[splitInfo.length - 1];
		}

		@Override
		public void open(Configuration parameters) {
			this.parallelism = getRuntimeContext().getNumberOfParallelSubtasks();
			coefSize = vectorSize;
		}

		@Override
		public void flatMap(Tuple4 <Long, Vector, Double, double[]> t4,
							Collector <Tuple6 <Long, Integer, Integer, Vector, Double, double[]>> collector) {
			Vector vec = t4.f1;

			if (vec instanceof SparseVector) {
				int[] indices = ((SparseVector) vec).indices;
				double[] values = ((SparseVector) vec).values;
				int pos = 1;
				int subNum = 0;
				boolean hasElement = false;

				for (int i = 0; i < indices.length; ++i) {
					if (indices[i] < splitInfo[pos]) {
						nnz[pos - 1]++;
						hasElement = true;
					} else {
						pos++;
						i--;
						if (hasElement) {
							subNum++;
							hasElement = false;
						}
					}
				}
				if (nnz[pos - 1] != 0) {
					subNum++;
				}
				pos = 0;
				for (int i = 0; i < nnz.length; ++i) {
					if (nnz[i] != 0) {
						int[] tmpIndices = new int[nnz[i]];
						double[] tmpValues = new double[nnz[i]];
						System.arraycopy(indices, pos, tmpIndices, 0, nnz[i]);
						System.arraycopy(values, pos, tmpValues, 0, nnz[i]);
						Vector tmpVec = new SparseVector(coefSize, tmpIndices, tmpValues);
						collector.collect(Tuple6.of(t4.f0, i, subNum, tmpVec, t4.f2, t4.f3));
						pos += nnz[i];
						nnz[i] = 0;
					}
				}
			} else {
				double[] data = ((DenseVector) vec).values;
				for (int i = 0; i < splitInfo.length - 1; ++i) {
					DenseVector dvec = new DenseVector(splitInfo[i + 1] - splitInfo[i]);
					if (splitInfo[i + 1] - splitInfo[i] >= 0) {
						System.arraycopy(data, splitInfo[i], dvec.values, 0,
							splitInfo[i + 1] - splitInfo[i]);
					}
					collector.collect(Tuple6.of(t4.f0, i, parallelism, dvec, t4.f2, t4.f3));
				}
			}
		}
	}

	public static class CalcTask extends RichFlatMapFunction <Tuple6 <Long, Integer, Integer, Vector, Double,
		double[]>,
		Tuple2 <Long, Object>> implements CheckpointedFunction {
		private static final long serialVersionUID = 1613267176484234752L;
		transient private double[] coef;
		transient private Map <Long, Tuple3 <Vector, Object, Long>> subVectors;

		private int startIdx;
		private int endIdx;
		private final int[] poses;
		long startTime = System.currentTimeMillis();
		int modelSaveTimeInterval;
		private long modelId = 0;
		private final FtrlLearningKernel kernel;
		private final int vectorSize;
		private int numWorkers;
		private int workerId;
		//private final boolean hasIntercept;
		private transient ListState <double[]> modelState;

		public CalcTask(int[] poses, FtrlLearningKernel kernel, int vectorSize) {
			this.poses = poses;
			this.modelSaveTimeInterval = 20; // todo read from params
			this.kernel = kernel;
			this.vectorSize = vectorSize;
		}

		@Override
		public void open(Configuration parameters) throws Exception {
			super.open(parameters);
			subVectors = new HashMap <>();
			startTime = System.currentTimeMillis();
		}

		@Override
		public void snapshotState(FunctionSnapshotContext context) throws Exception {
			double[] state = coef;
			modelState.clear();
			modelState.add(state);
		}

		@Override
		public void initializeState(FunctionInitializationContext context) throws Exception {
			modelState =
				context.getOperatorStateStore()
					.getListState(
						new ListStateDescriptor <>(
							"statStreamingOnlineFtrlModelState",
							TypeInformation.of(new TypeHint <double[]>() {
							})));

			numWorkers = getRuntimeContext().getNumberOfParallelSubtasks();
			workerId = getRuntimeContext().getIndexOfThisSubtask();
			startIdx = poses[workerId];
			endIdx = poses[workerId + 1];
			if (context.isRestored()) {
				for (double[] state : modelState.get()) {
					this.coef = state;
					int localSize = coef.length;
					kernel.setModelParams(localSize);
				}
			}
		}

		@Override
		public void flatMap(
			Tuple6 <Long, Integer, Integer, Vector, Double, double[]> value,
			Collector <Tuple2 <Long, Object>> out) throws InterruptedException {
			Thread.sleep(1000);

			if (this.coef == null) {
				LinearModelData logisticRegressionModelData =
					(LinearModelData)
						getRuntimeContext().getBroadcastVariable("initModel").get(0);
				DenseVector coefVector = logisticRegressionModelData.coefficient;
				int localSize = vectorSize / numWorkers;
				localSize += (workerId < vectorSize % numWorkers) ? 1 : 0;
				kernel.setModelParams(localSize);

				coef = new double[localSize];
				endIdx = Math.min(endIdx, coefVector.size());
				for (int i = startIdx; i < endIdx; ++i) {
					coef[i - startIdx] = coefVector.get(i);
				}
			}

			Long timeStamps = System.currentTimeMillis();
			if (value.f0 >= 0) {
				Vector vec = value.f3;
				double[] rval = kernel.calcLocalWx(coef, vec, startIdx);
				subVectors.put(value.f0, Tuple3.of(vec, value.f4, timeStamps));
				out.collect(Tuple2.of(value.f0, Tuple2.of(value.f2, rval)));
			} else {
				Tuple3 <Vector, Object, Long> t3 = subVectors.get(-(value.f0 + 1));
				long timeInterval = timeStamps - t3.f2;
				Vector vec = t3.f0;
				double[] r = value.f5;
				kernel.updateModel(coef, vec, r, timeInterval, startIdx, value.f4);
				subVectors.remove(-(value.f0 + 1));
				if (System.currentTimeMillis() - startTime > modelSaveTimeInterval) {
					startTime = System.currentTimeMillis();
					modelId++;
					out.collect(Tuple2.of(-modelId,
						Tuple2.of(getRuntimeContext().getIndexOfThisSubtask(), coef)));
				}
			}
		}
	}

	public static class ReduceTask extends
		RichFlatMapFunction <Tuple2 <Long, Object>, Tuple2 <Long, Object>> {
		private static final long serialVersionUID = 1072071076831105639L;
		private final int parallelism;
		private final int[] poses;
		transient private Map <Long, Tuple2 <Integer, double[]>> buffer;
		FtrlLearningKernel kernel;
		private Map <Long, List <Tuple2 <Integer, double[]>>> models;

		public ReduceTask(int parallelism, int[] poses, FtrlLearningKernel kernel) {
			this.parallelism = parallelism;
			this.poses = poses;
			this.kernel = kernel;
		}

		@Override
		public void open(Configuration parameters) throws Exception {
			super.open(parameters);
			buffer = new HashMap <>(0);
			models = new HashMap <>(0);
		}

		@Override
		public void flatMap(Tuple2 <Long, Object> value,
							Collector <Tuple2 <Long, Object>> out) throws InterruptedException {
			if (value.f0 < 0) {
				long modelId = value.f0;
				Tuple2 <Integer, double[]> t2 = (Tuple2 <Integer, double[]>) value.f1;
				List <Tuple2 <Integer, double[]>> model = models.get(modelId);
				if (model == null) {
					model = new ArrayList <>();
					model.add(Tuple2.of(t2.f0, t2.f1));
					models.put(modelId, model);
				} else {
					model.add(Tuple2.of(t2.f0, t2.f1));
				}
				if (model.size() == parallelism) {
					double[] coef = new double[poses[parallelism]];
					for (Tuple2 <Integer, double[]> subModel : model) {
						int pos = poses[subModel.f0];
						System.arraycopy(subModel.f1, 0, coef, pos, subModel.f1.length);
					}
					out.collect(Tuple2.of(value.f0, coef));
					models.remove(modelId);
				}
			} else {
				Tuple2 <Integer, double[]> val = buffer.get(value.f0);
				Tuple2 <Integer, double[]> t2 = (Tuple2 <Integer, double[]>) value.f1;
				if (val == null) {
					val = Tuple2.of(1, t2.f1);
					buffer.put(value.f0, val);
				} else {
					val.f0++;
					for (int i = 0; i < val.f1.length; ++i) {
						val.f1[i] += t2.f1[i];
					}
				}

				if (val.f0.equals(t2.f0)) {
					out.collect(Tuple2.of(value.f0, kernel.getFeedbackVar(val.f1)));
					buffer.remove(value.f0);
				}
				Thread.sleep(1000);
			}
		}
	}

	private static class SubVectorPartitioner implements Partitioner <Integer> {

		private static final long serialVersionUID = 3154122892861557361L;

		@Override
		public int partition(Integer key, int numPartitions) {
			return key % numPartitions;
		}
	}

	private static class WxPartitioner implements Partitioner <Long> {

		private static final long serialVersionUID = -6637943571982178520L;

		@Override
		public int partition(Long sampleId, int numPartition) {
			if (sampleId < 0L) {
				return 0;
			} else {
				return (int) (sampleId % numPartition);
			}
		}
	}

	public static class FtrlLearningKernel implements Serializable {
		private double alpha;
		private double beta;
		private double l1;
		private double l2;
		private double[] nParam;
		private double[] zParam;

		public void setModelParams(int localModelSize) {
			nParam = new double[localModelSize];
			zParam = new double[localModelSize];
			this.alpha = 0.1;
			this.beta = 0.1;
			this.l1 = 0.1;
			this.l2 = 0.1;
		}

		public double[] getFeedbackVar(double[] wx) {
			return new double[] {1 / (1 + Math.exp(-wx[0]))};
		}

		public double[] calcLocalWx(double[] coef, Vector vec, int startIdx) {
			double y = 0.0;
			if (vec instanceof SparseVector) {
				int[] indices = ((SparseVector) vec).indices;

				for (int i = 0; i < indices.length; ++i) {
					y += ((SparseVector) vec).values[i] * coef[indices[i] - startIdx];
				}
			} else {
				for (int i = 0; i < vec.size(); ++i) {
					y += vec.get(i) * coef[i];
				}
			}
			return new double[] {y};
		}

		public void updateModel(double[] coef, Vector vec, double[] wx, long timeInterval,
								int startIdx, double labelValue) {
			double pred = wx[0];

			if (vec instanceof SparseVector) {
				int[] indices = ((SparseVector) vec).indices;
				double[] values = ((SparseVector) vec).values;

				for (int i = 0; i < indices.length; ++i) {
					// update zParam nParam
					int id = indices[i] - startIdx;
					double g = (pred - labelValue) * values[i] / Math.sqrt(timeInterval + 1);
					double sigma = (Math.sqrt(nParam[id] + g * g) - Math.sqrt(nParam[id])) / alpha;
					zParam[id] += g - sigma * coef[id];
					nParam[id] += g * g;

					// update model coefficient
					if (Math.abs(zParam[id]) <= l1) {
						coef[id] = 0.0;
					} else {
						coef[id] = ((zParam[id] < 0 ? -1 : 1) * l1 - zParam[id])
							/ ((beta + Math.sqrt(nParam[id]) / alpha + l2));
					}
				}
			} else {
				double[] data = ((DenseVector) vec).values;

				for (int i = 0; i < data.length; ++i) {
					// update zParam nParam
					double g = (pred - labelValue) * data[i] / Math.sqrt(timeInterval + 1);
					double sigma = (Math.sqrt(nParam[i] + g * g) - Math.sqrt(nParam[i])) / alpha;
					zParam[i] += g - sigma * coef[i];
					nParam[i] += g * g;

					// update model coefficient
					if (Math.abs(zParam[i]) <= l1) {
						coef[i] = 0.0;
					} else {
						coef[i] = ((zParam[i] < 0 ? -1 : 1) * l1 - zParam[i])
							/ ((beta + Math.sqrt(nParam[i]) / alpha + l2));
					}
				}
			}
		}
	}

	@Override
	public void save(String path) {

	}

	@Override
	public Map <Param <?>, Object> getParamMap() {
		return paramMap;
	}
}
