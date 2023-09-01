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

package org.apache.flink.ml.common.datastream;

import org.apache.flink.annotation.Internal;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.typeutils.TypeExtractor;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.datastream.purefunc.MapWithBcPureFunc;
import org.apache.flink.ml.common.datastream.purefunc.PureFuncContextImpl;
import org.apache.flink.ml.common.datastream.purefunc.RichPureFunc;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.AbstractUdfStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.TimestampedCollector;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

/** Provides utility functions for {@link DataStream} in iterations. */
@Internal
public class DataStreamInIterationUtils {
    public static <IN, OUT> DataStream<OUT> mapPartition(
            DataStream<IN> input, MapPartitionFunction<IN, OUT> func) {
        TypeInformation<OUT> outType =
                TypeExtractor.getMapPartitionReturnTypes(func, input.getType(), null, true);
        return mapPartition(input, func, outType);
    }

    public static <IN, OUT> DataStream<OUT> mapPartition(
            DataStream<IN> input,
            MapPartitionFunction<IN, OUT> func,
            TypeInformation<OUT> outType) {
        return input.transform(
                        "mapPartition", outType, new MapPartitionOperator<>(func, input.getType()))
                .setParallelism(input.getParallelism());
    }

    public static <IN, OUT, BC> DataStream<OUT> mapWithBc(
            DataStream<IN> input,
            DataStream<BC> broadcast,
            MapWithBcPureFunc<IN, OUT, BC> func,
            TypeSerializer<IN> inTypeSerializer,
            TypeInformation<OUT> outType) {
        return input.connect(broadcast.broadcast())
                .transform(
                        "mapWithBroadcastInIteration",
                        outType,
                        new MapWithBcOperator<>(inTypeSerializer, func));
    }

    public static <T> DataStream<T> reduce(DataStream<T> input, ReduceFunction<T> func) {
        return reduce(input, func, input.getType());
    }

    public static <T> DataStream<T> reduce(
            DataStream<T> input, ReduceFunction<T> func, TypeInformation<T> typeInfo) {
        TypeSerializer<T> typeSerializer = typeInfo.createSerializer(input.getExecutionConfig());
        DataStream<T> partialReducedStream =
                input.transform(
                                "reduceInIteration",
                                typeInfo,
                                new ReduceOperator<>(func, typeSerializer))
                        .setParallelism(input.getParallelism());
        if (partialReducedStream.getParallelism() == 1) {
            return partialReducedStream;
        } else {
            return partialReducedStream
                    .transform(
                            "reduceInIteration",
                            typeInfo,
                            new ReduceOperator<>(func, typeSerializer))
                    .setParallelism(1);
        }
    }

    /**
     * A stream operator to apply {@link MapPartitionFunction} on each partition of the input
     * bounded data stream.
     */
    private static class MapPartitionOperator<IN, OUT>
            extends AbstractUdfStreamOperator<OUT, MapPartitionFunction<IN, OUT>>
            implements OneInputStreamOperator<IN, OUT>, IterationListener<OUT> {

        private final TypeInformation<IN> type;
        private ListStateWithCache<IN> valuesState;

        public MapPartitionOperator(MapPartitionFunction<IN, OUT> func, TypeInformation<IN> type) {
            super(func);
            this.type = type;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            valuesState =
                    new ListStateWithCache<>(
                            type.createSerializer(getExecutionConfig()),
                            getContainingTask(),
                            getRuntimeContext(),
                            context,
                            config.getOperatorID());
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            valuesState.snapshotState(context);
        }

        @Override
        public void processElement(StreamRecord<IN> input) throws Exception {
            valuesState.add(input.getValue());
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<OUT> collector) throws Exception {
            userFunction.mapPartition(valuesState.get(), new TimestampedCollector<>(output));
            valuesState.clear();
        }

        @Override
        public void onIterationTerminated(Context context, Collector<OUT> collector) {}
    }

    static class MapWithBcOperator<IN, OUT, BC> extends AbstractStreamOperator<OUT>
            implements TwoInputStreamOperator<IN, BC, OUT>, IterationListener<OUT> {
        private final TypeSerializer<IN> inTypeSerializer;
        private final MapWithBcPureFunc<IN, OUT, BC> func;
        private BC bc;
        private ListStateWithCache<IN> cachedIn;

        MapWithBcOperator(
                TypeSerializer<IN> inTypeSerializer, MapWithBcPureFunc<IN, OUT, BC> func) {
            this.inTypeSerializer = inTypeSerializer;
            this.func = func;
        }

        @Override
        public void open() throws Exception {
            super.open();
            if (func instanceof RichPureFunc) {
                ((RichPureFunc) func)
                        .setContext(
                                new PureFuncContextImpl(
                                        getRuntimeContext().getNumberOfParallelSubtasks(),
                                        getRuntimeContext().getIndexOfThisSubtask()));
                ((RichPureFunc) func).open();
            }
        }

        @Override
        public void close() throws Exception {
            if (func instanceof RichPureFunc) {
                ((RichPureFunc) func).close();
            }
            super.close();
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            cachedIn =
                    new ListStateWithCache<>(
                            inTypeSerializer,
                            getContainingTask(),
                            getRuntimeContext(),
                            context,
                            getOperatorID());
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            cachedIn.snapshotState(context);
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<OUT> collector) throws Exception {
            Preconditions.checkNotNull(bc);
            for (IN value : cachedIn.get()) {
                OUT out = func.map(value, bc);
                collector.collect(out);
            }
            if (func instanceof RichPureFunc) {
                ((RichPureFunc) func).close();
                ((RichPureFunc) func).open();
            }
            bc = null;
            cachedIn.clear();
        }

        @Override
        public void onIterationTerminated(Context context, Collector<OUT> collector) {}

        @Override
        public void processElement1(StreamRecord<IN> element) throws Exception {
            if (null == bc) {
                cachedIn.add(element.getValue());
                return;
            }
            for (IN value : cachedIn.get()) {
                OUT out = func.map(value, bc);
                output.collect(new StreamRecord<>(out));
            }
            cachedIn.clear();
            OUT out = func.map(element.getValue(), bc);
            output.collect(new StreamRecord<>(out));
        }

        @Override
        public void processElement2(StreamRecord<BC> element) {
            bc = element.getValue();
        }
    }

    private static class ReduceOperator<T> extends AbstractUdfStreamOperator<T, ReduceFunction<T>>
            implements OneInputStreamOperator<T, T>, IterationListener<T> {
        private final TypeSerializer<T> typeSerializer;
        /** The temp result of the reduce function. */
        private T result;

        private ListState<T> state;

        public ReduceOperator(ReduceFunction<T> userFunction, TypeSerializer<T> typeSerializer) {
            super(userFunction);
            this.typeSerializer = typeSerializer;
        }

        @Override
        public void processElement(StreamRecord<T> streamRecord) throws Exception {
            if (result == null) {
                result = streamRecord.getValue();
            } else {
                result = userFunction.reduce(streamRecord.getValue(), result);
            }
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            state =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("state", typeSerializer));
            result = OperatorStateUtils.getUniqueElement(state, "state").orElse(null);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            state.clear();
            if (result != null) {
                state.add(result);
            }
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<T> collector) {
            if (result != null) {
                output.collect(new StreamRecord<>(result));
                result = null;
            }
        }

        @Override
        public void onIterationTerminated(Context context, Collector<T> collector) {}
    }
}
