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

package org.apache.flink.ml.common.ps.api;

import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.ps.iterations.HighComponent;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

/** Estimator Stage. */
public class EstimatorComponent extends HighComponent {

    public transient Estimator<?, ?> estimator;
    public boolean isOutputModelData = false;

    public EstimatorComponent(Estimator<?, ?> estimator) {
        this.estimator = estimator;
    }

    public EstimatorComponent(Estimator<?, ?> estimator, boolean isOutputModelData) {
        this.estimator = estimator;
        this.isOutputModelData = isOutputModelData;
    }

    @Override
    public MLData apply(MLData mlData) {
        if (isForEachRound) {
            DataStream<?> output =
                    IterationBody.forEachRound(
                                    DataStreamList.of(mlData.get(fromName)),
                                    input -> {
                                        StreamTableEnvironment tEnv =
                                                StreamTableEnvironment.create(
                                                        input.get(0).getExecutionEnvironment());
                                        Table table = tEnv.fromDataStream(input.get(0));
                                        Model<?> model = this.estimator.fit(table);

                                        DataStream<?> feedback =
                                                tEnv.toDataStream(model.transform(table)[0]);
                                        return DataStreamList.of(feedback);
                                    })
                            .get(0);
            mlData.add(toName, output);
        } else {
            Table table = mlData.getTable(fromName);
            Model<?> model = this.estimator.fit(table);
            mlData.add(toName, model.transform(table)[0]);
        }
        mlData.setCurrentProcessName(toName);

        return mlData;
    }
}
