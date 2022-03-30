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
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.classification.ftrl.Ftrl;
import org.apache.flink.ml.classification.knn.Knn;
import org.apache.flink.ml.classification.knn.KnnModel;
import org.apache.flink.ml.classification.logisticregression.LogisticRegression;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.util.InMemorySourceFunction;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Tests {@link Knn} and {@link KnnModel}.
 */
public class FtrlTest {
	@Rule
	public final TemporaryFolder tempFolder = new TemporaryFolder();
	private StreamExecutionEnvironment env;
	private StreamTableEnvironment tEnv;
	private Table trainData;
	private Table onlineTrainTable;

	private static final List <Row> trainRows = new ArrayList <>();

	private InMemorySourceFunction <Row> trainSource;

	@Before
	public void before() {
		Configuration config = new Configuration();
		config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
		env = StreamExecutionEnvironment.getExecutionEnvironment(config);
		env.setParallelism(1);
		env.setRestartStrategy(RestartStrategies.noRestart());
		env.enableCheckpointing(10000000, CheckpointingMode.AT_LEAST_ONCE, true);
		tEnv = StreamTableEnvironment.create(env);
		trainSource = new InMemorySourceFunction<>();
		Random rand = new Random();
		for (int i = 0; i < 100; ++i) {
			trainRows.add(Row.of(1.0*(i%2), new DenseVector(new double[] {rand.nextGaussian(), rand.nextGaussian()})));
		}
		Schema schema =
			Schema.newBuilder()
				.column("f0", DataTypes.of(DenseVector.class))
				.column("f1", DataTypes.DOUBLE())
				.build();
		DataStream <Row> dataStream = env.fromCollection(trainRows);
		trainData = tEnv.fromDataStream(dataStream, schema).as("features", "label");

		onlineTrainTable =
			tEnv.fromDataStream(env.addSource(trainSource, new RowTypeInfo(new TypeInformation[] {
					TypeInformation.of(DenseVector.class),
					Types.DOUBLE
				},
				new String[] {"features", "label"})));
	}

	@Test
	public void testFtrl() throws Exception {
		Table initModel = new LogisticRegression()
			.setFeaturesCol("features").setLabelCol("label").fit(trainData).getModelData()[0];

		Table models = new Ftrl()
			.setFeaturesCol("features").setLabelCol("label").setVectorSize(2).fit(initModel, trainData).getModelData()[0];

		DataStream<Row> result = tEnv.toDataStream(models)
			.map((MapFunction <Row, Row>) value -> {
				System.out.println(value);
				return value;
			});
		result.executeAndCollect();
	}
}
