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

import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.ps.iterations.HighComponent;
import org.apache.flink.streaming.api.datastream.DataStream;

/** Map Stage. */
public class MiniBatchComponent extends HighComponent {

    private final int batchSize;

    public MiniBatchComponent(int batchSize) {
        this.batchSize = batchSize;
    }

    @Override
    public MLData apply(MLData mlData) {
        DataStream<?> input = mlData.get(fromName);

        mlData.add(
                toName,
                DataStreamUtils.generateBatchData(input, input.getParallelism(), batchSize));

        mlData.setCurrentProcessName(toName);
        return mlData;
    }
}
