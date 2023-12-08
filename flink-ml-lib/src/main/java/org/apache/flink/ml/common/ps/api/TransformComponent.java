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
import org.apache.flink.ml.common.ps.iterations.CommonComponent;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;

/** FlatMap Stage. */
public abstract class TransformComponent<IN, R> extends AbstractStreamOperator<R>
        implements CommonComponent, OneInputStreamOperator<IN, R>, BoundedOneInput {
    protected String fromName = null;
    protected String toName = null;
    private int parallel = -1;

    private TypeInformation<?> type;

    public TransformComponent<IN, R> input(String fromName) {
        this.fromName = fromName;
        return this;
    }

    public TransformComponent<IN, R> output(String toName) {
        this.toName = toName;
        return this;
    }

    public TransformComponent<IN, R> returns(TypeInformation<?> type) {
        this.type = type;
        return this;
    }

    @Override
    public void endInput() throws Exception {}

    public TransformComponent<IN, R> withParallel(int parallel) {
        this.parallel = parallel;
        return this;
    }

    @Override
    @SuppressWarnings("unchecked")
    public MLData apply(MLData mlData) {

        parallel = parallel == -1 ? mlData.get(fromName).getParallelism() : parallel;
        DataStream<?> dataStream =
                mlData.get(fromName)
                        .transform(this.getClass().getSimpleName(), type, this)
                        .setParallelism(parallel);
        mlData.add(toName, dataStream);
        mlData.setCurrentProcessName(toName);
        return mlData;
    }
}
