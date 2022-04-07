/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.classification;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.classification.ftrl.Ftrl;
import org.apache.flink.ml.classification.ftrl.FtrlModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static junit.framework.TestCase.assertEquals;

/**
 * Tests {@link Ftrl} and {@link FtrlModel}.
 */
public class FtrlTest {
	@Rule
	public final TemporaryFolder tempFolder = new TemporaryFolder();
	private StreamExecutionEnvironment env;
	private StreamTableEnvironment tEnv;

	private static final List <Row> TRAIN_DENSE_ROWS =
		Arrays.asList(
			Row.of(Vectors.dense(0.1, 2.), 0.),
			Row.of(Vectors.dense(0.2, 2.), 0.),
			Row.of(Vectors.dense(0.3, 2.), 0.),
			Row.of(Vectors.dense(0.4, 2.), 0.),
			Row.of(Vectors.dense(0.5, 2.), 0.),
			Row.of(Vectors.dense(11., 12.), 1.),
			Row.of(Vectors.dense(12., 11.), 1.),
			Row.of(Vectors.dense(13., 12.), 1.),
			Row.of(Vectors.dense(14., 12.), 1.),
			Row.of(Vectors.dense(15., 12.), 1.));

	private static final List <Row> TRAIN_SPARSE_ROWS =
		Arrays.asList(
			Row.of(Vectors.sparse(10, new int[] {1, 3, 4}, new double[] {1.0, 1.0, 1.0}), 0.),
			Row.of(Vectors.sparse(10, new int[] {0, 2, 3}, new double[] {1.0, 1.0, 1.0}), 0.),
			Row.of(Vectors.sparse(10, new int[] {0, 1, 3, 4}, new double[] {1.0, 1.0, 1.0, 1.0}), 0.),
			Row.of(Vectors.sparse(10, new int[] {2, 3, 4}, new double[] {1.0, 1.0, 1.0}), 0.),
			Row.of(Vectors.sparse(10, new int[] {1, 3, 4}, new double[] {1.0, 1.0, 1.0}), 0.),
			Row.of(Vectors.sparse(10, new int[] {6, 7, 8}, new double[] {1.0, 1.0, 1.0}), 1.),
			Row.of(Vectors.sparse(10, new int[] {6, 8, 9}, new double[] {1.0, 1.0, 1.0}), 1.),
			Row.of(Vectors.sparse(10, new int[] {5, 8, 9}, new double[] {1.0, 1.0, 1.0}), 1.),
			Row.of(Vectors.sparse(10, new int[] {5, 6, 8}, new double[] {1.0, 1.0, 1.0}), 1.),
			Row.of(Vectors.sparse(10, new int[] {5, 6, 7}, new double[] {1.0, 1.0, 1.0}), 1.));

	private static final List <Row> PREDICT_DENSE_ROWS =
		Arrays.asList(
			Row.of(Vectors.dense(0.8, 2.7), 0.),
			Row.of(Vectors.dense(0.8, 2.4), 0.),
			Row.of(Vectors.dense(0.7, 2.3), 0.),
			Row.of(Vectors.dense(0.4, 2.7), 0.),
			Row.of(Vectors.dense(0.5, 2.8), 0.),
			Row.of(Vectors.dense(10.2, 12.1), 1.),
			Row.of(Vectors.dense(13.3, 13.1), 1.),
			Row.of(Vectors.dense(13.5, 12.2), 1.),
			Row.of(Vectors.dense(14.9, 12.5), 1.),
			Row.of(Vectors.dense(15.5, 11.2), 1.));

	private static final List <Row> PREDICT_SPARSE_ROWS =
		Arrays.asList(
			Row.of(Vectors.sparse(10, new int[] {1, 2, 4}, new double[] {1.0, 1.0, 1.0}), 0.),
			Row.of(Vectors.sparse(10, new int[] {2, 3, 4}, new double[] {1.0, 1.0, 1.0}), 0.),
			Row.of(Vectors.sparse(10, new int[] {0, 1, 2, 4}, new double[] {1.0, 1.0, 1.0, 1.0}), 0.),
			Row.of(Vectors.sparse(10, new int[] {1, 3, 4}, new double[] {1.0, 1.0, 1.0}), 0.),
			Row.of(Vectors.sparse(10, new int[] {6, 7, 9}, new double[] {1.0, 1.0, 1.0}), 1.),
			Row.of(Vectors.sparse(10, new int[] {7, 8, 9}, new double[] {1.0, 1.0, 1.0}), 1.),
			Row.of(Vectors.sparse(10, new int[] {5, 7, 9}, new double[] {1.0, 1.0, 1.0}), 1.),
			Row.of(Vectors.sparse(10, new int[] {5, 6, 7}, new double[] {1.0, 1.0, 1.0}), 1.),
			Row.of(Vectors.sparse(10, new int[] {5, 8, 9}, new double[] {1.0, 1.0, 1.0}), 1.));

	@Before
	public void before() throws Exception {
		Configuration config = new Configuration();
		config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
		env = StreamExecutionEnvironment.getExecutionEnvironment(config);
		env.setParallelism(4);
		env.enableCheckpointing(100);
		env.setRestartStrategy(RestartStrategies.noRestart());
		tEnv = StreamTableEnvironment.create(env);
	}

	@Test
	public void testFtrl() throws Exception {
		Table initModel = tEnv.fromDataStream(env.fromElements(Row.of(new DenseVector(new double[] {0.0, 0.0}), 0L)));
		Table onlinePredictTable =
			tEnv.fromDataStream(
				env.addSource(new RandomSourceFunction(10, 1000, TRAIN_DENSE_ROWS), new RowTypeInfo(new TypeInformation[] {
					TypeInformation.of(DenseVector.class),
					Types.DOUBLE
				}, new String[] {"features", "label"})));

		Table models = new Ftrl()
			.setFeaturesCol("features").setInitialModelData(initModel).setGlobalBatchSize(100).setLabelCol("label").fit
				(onlinePredictTable).getModelData()[0];
		tEnv.toDataStream(models).print();
		env.execute();
	}

	@Test
	public void testFtrlModel() throws Exception {
		Table initModel = tEnv.fromDataStream(env.fromElements(Row.of(new DenseVector(new double[] {0.0, 0.0}), 0L)));

		Table onlineTrainTable =
			tEnv.fromDataStream(
				env.addSource(new RandomSourceFunction(1, 10000, TRAIN_DENSE_ROWS), new RowTypeInfo(new TypeInformation[] {
					TypeInformation.of(Vector.class),
					Types.DOUBLE
				}, new String[] {"features", "label"})));
		Table onlinePredictTable =
			tEnv.fromDataStream(
				env.addSource(new RandomSourceFunction(200, 100, PREDICT_DENSE_ROWS), new RowTypeInfo(new TypeInformation[] {
					TypeInformation.of(Vector.class),
					Types.DOUBLE
				}, new String[] {"features", "label"})));

		FtrlModel model = new Ftrl()
			.setFeaturesCol("features").setInitialModelData(initModel).setGlobalBatchSize(1000).setLabelCol("label").fit
				(onlineTrainTable);
		Table resultTable = model.setPredictionCol("pred").transform(onlinePredictTable)[0];
		tEnv.toDataStream(model.getModelData()[0]).print();
		tEnv.toDataStream(resultTable).print();
		verifyPredictionResult(resultTable, "label", "pred", "modelVersion");
	}

	@Test
	public void testFtrlModelSparse() throws Exception {
		Table initModelSparse = tEnv.fromDataStream(
			env.fromElements(Row.of(new DenseVector(new double[] {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}), 0L)));

		Table onlineTrainTable =
			tEnv.fromDataStream(
				env.addSource(new RandomSourceFunction(10, 2000, TRAIN_SPARSE_ROWS), new RowTypeInfo(new TypeInformation[] {
					TypeInformation.of(Vector.class),
					Types.DOUBLE
				}, new String[] {"features", "label"})));
		Table onlinePredictTable =
			tEnv.fromDataStream(
				env.addSource(new RandomSourceFunction(10, 2000, PREDICT_SPARSE_ROWS), new RowTypeInfo(new TypeInformation[] {
					TypeInformation.of(Vector.class),
					Types.DOUBLE
				}, new String[] {"features", "label"})));

		FtrlModel model = new Ftrl()
			.setFeaturesCol("features").setInitialModelData(initModelSparse).setGlobalBatchSize(100).setLabelCol(
				"label").fit
				(onlineTrainTable);
		Table resultTable = model.setPredictionCol("pred").transform(onlinePredictTable)[0];
		//tEnv.toDataStream(resultTable).print();
		verifyPredictionResult(resultTable, "label", "pred", "modelVersion");
		//tEnv.toDataStream(model.getModelData()[0]).print();
		//tEnv.toDataStream(resultTable).print();
		//env.execute();
	}

	private static void verifyPredictionResult(Table output, String labelCol, String predictionCol, String modelVersion)
		throws Exception {
		StreamTableEnvironment tEnv =
			(StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
		DataStream <Tuple3 <Double, Double, Long>> stream =
			tEnv.toDataStream(output)
				.map(
					new MapFunction <Row, Tuple3 <Double, Double, Long>>() {
						@Override
						public Tuple3<Double, Double, Long> map(Row row) {
							return Tuple3.of(
								(Double) row.getField(labelCol),
								(Double) row.getField(predictionCol),
								(Long) row.getField(modelVersion));
						}
					});
		List<Tuple3<Double, Double, Long>> result = IteratorUtils.toList(stream.executeAndCollect());
		Map <Long, Tuple2<Double, Double>> correctRatio = new HashMap <>();

		for (Tuple3<Double, Double, Long> t3 : result) {
			if (correctRatio.containsKey(t3.f2)) {
				Tuple2<Double, Double> t2 = correctRatio.get(t3.f2);
				if (t3.f0.equals(t3.f1)) {
					t2.f0 += 1.0;
				}
				t2.f1 += 1.0;
			} else {
				correctRatio.put(t3.f2, Tuple2.of(t3.f0.equals(t3.f1) ? 1.0 : 0.0, 1.0));
			}

		}
		for (Long id : correctRatio.keySet()) {
			if (id > 0L) {
				assertEquals(1.0, correctRatio.get(id).f0 / correctRatio.get(id).f1, 1.0e-5);
			}
		}
	}

	public static class RandomSourceFunction implements SourceFunction <Row> {
		private volatile boolean isRunning = true;
		private final long timeInterval;
		private final long maxSize;
		private final List <Row> data;

		public RandomSourceFunction(long timeInterval, long maxSize, List <Row> data) {
			this.timeInterval = timeInterval;
			this.maxSize = maxSize;
			this.data = data;
		}

		@Override
		public void run(SourceContext <Row> ctx) throws Exception {
			int size = data.size();
			for (int i = 0; i < maxSize; ++i) {
				int idx = i % size;
				if (isRunning) {
					ctx.collect(data.get(idx));
					if (timeInterval > 0) {
						Thread.sleep(timeInterval);
					}
				}
			}
		}

		@Override
		public void cancel() {
			isRunning = false;
		}
	}
}
