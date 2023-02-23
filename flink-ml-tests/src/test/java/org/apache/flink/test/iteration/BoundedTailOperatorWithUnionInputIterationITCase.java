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

package org.apache.flink.test.iteration;

import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationConfig.OperatorLifeCycle;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.common.iteration.TerminateOnMaxIter;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/** Tests the case that Tail operator contains multiple inputs. */
public class BoundedTailOperatorWithUnionInputIterationITCase {

    @Test
    public void testTailOperatorWithUnionInput() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        DataStream<Long> input1 = env.fromSequence(0, 10);
        DataStream<Long> input2 = env.fromSequence(0, 10);
        String expectedErrorMessage =
                "Tail operator should have only one input. Please check whether operator "
                        + "\"Union\" contains multiple inputs.";

        try {
            Iterations.iterateBoundedStreamsUntilTermination(
                    DataStreamList.of(input1),
                    ReplayableDataStreamList.replay(input2),
                    new IterationConfig(OperatorLifeCycle.PER_ROUND),
                    new IterationBodyWithUnionAsFeedback());
        } catch (UnsupportedOperationException e) {
            assertEquals(expectedErrorMessage, e.getMessage());
        }
    }

    private static class IterationBodyWithUnionAsFeedback implements IterationBody {

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {

            DataStream<Long> variableStream = variableStreams.get(0);
            DataStream<Long> constantStream = dataStreams.get(0);
            DataStream<Integer> terminationCriteria =
                    constantStream.flatMap(new TerminateOnMaxIter<>(3));
            return new IterationBodyResult(
                    DataStreamList.of(variableStream.union(constantStream)),
                    DataStreamList.of(variableStream),
                    terminationCriteria);
        }
    }
}
