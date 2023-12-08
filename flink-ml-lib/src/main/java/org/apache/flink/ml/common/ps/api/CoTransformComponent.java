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

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.ml.common.ps.iterations.CommonComponent;
import org.apache.flink.streaming.api.datastream.ConnectedStreams;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

/** FlatMap Stage. */
public abstract class CoTransformComponent<IN1, IN2, R> extends AbstractStreamOperator<R>
        implements TwoInputStreamOperator<IN1, IN2, R>, IterationListener<R>, CommonComponent {

    private TypeInformation<?> type;

    public OutputTag outputTag;

    public CoTransformComponent<IN1, IN2, R> withOutType(TypeInformation<?> type) {
        this.type = type;
        return this;
    }

    protected String fromName = null;
    protected String toName = null;

    private String connectedName;

    public CoTransformComponent() {}

    public CoTransformComponent<IN1, IN2, R> input(String fromName) {
        this.fromName = fromName;
        return this;
    }

    public CoTransformComponent<IN1, IN2, R> with(String connectedName) {
        this.connectedName = connectedName;
        return this;
    }

    public CoTransformComponent<IN1, IN2, R> withOutputTag(OutputTag<?> outputTag) {
        this.outputTag = outputTag;
        return this;
    }

    public CoTransformComponent<IN1, IN2, R> output(String toName) {
        this.toName = toName;
        return this;
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<R> collector) throws Exception {}

    @Override
    public void onIterationTerminated(Context context, Collector<R> collector) throws Exception {}

    @Override
    @SuppressWarnings("unchecked")
    public MLData apply(MLData mlData) {
        ConnectedStreams connectedStreams = mlData.get(fromName).connect(mlData.get(connectedName));

        DataStream<?> dataStream =
                connectedStreams.transform(this.getClass().getSimpleName(), type, this);
        mlData.add(toName, dataStream);
        mlData.setCurrentProcessName(toName);
        return mlData;
    }

    public CoTransformComponent<IN1, IN2, R> returns(TypeInformation<?> type) {
        this.type = type;
        return this;
    }
}
