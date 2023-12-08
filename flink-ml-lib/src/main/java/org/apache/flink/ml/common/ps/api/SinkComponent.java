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

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.connector.file.sink.FileSink;
import org.apache.flink.core.fs.Path;
import org.apache.flink.ml.common.ps.iterations.BaseComponent;
import org.apache.flink.streaming.api.functions.sink.filesystem.bucketassigners.BasePathBucketAssigner;
import org.apache.flink.streaming.api.functions.sink.filesystem.rollingpolicies.OnCheckpointRollingPolicy;
import org.apache.flink.types.Row;

import java.io.IOException;
import java.io.OutputStream;

/** Common stage. */
public class SinkComponent implements BaseComponent {

    protected String from = null;
    protected String filePath = null;

    public final SinkComponent input(String input) {
        this.from = input;
        return this;
    }

    public final SinkComponent filePath(String filePath) {
        this.filePath = filePath;
        return this;
    }

    public void sink(MLData mlData) {
        FileSink<Row> sink =
                FileSink.forRowFormat(new Path(filePath), new RowEncoder())
                        .withRollingPolicy(OnCheckpointRollingPolicy.build())
                        .withBucketAssigner(new BasePathBucketAssigner<>())
                        .build();
        mlData.get(from).sinkTo(sink);
    }

    /** Comments. */
    public static class RowEncoder implements Encoder<Row> {

        @Override
        public void encode(Row row, OutputStream outputStream) throws IOException {

            StringBuilder stringBuilder = new StringBuilder();
            for (int i = 0; i < row.getArity(); ++i) {
                stringBuilder.append(row.getField(i));
                if (i != row.getArity() - 1) {
                    stringBuilder.append(",");
                }
            }
            stringBuilder.append("\n");
            outputStream.write(stringBuilder.toString().getBytes());
        }
    }
}
