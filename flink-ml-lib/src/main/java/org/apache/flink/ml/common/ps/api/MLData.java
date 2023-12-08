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

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.Function;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.typeutils.TypeExtractor;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.EndOfStreamWindows;
import org.apache.flink.ml.common.ps.iterations.CommonComponent;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;

import org.apache.commons.collections.IteratorUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Machine Learning Data. */
public class MLData {

    private final List<DataStream<?>> dataStreams = new ArrayList<>();
    private final Map<String, Integer> nameToIndex = new HashMap<>();

    private int currentProcessId = 0;

    public MLData(DataStream<?>[] streams, String[] names) {
        for (int i = 0; i < streams.length; ++i) {

            dataStreams.add(streams[i]);
            nameToIndex.put(names[i], i);
        }
    }

    public MLData(Table[] tables, String[] names) {
        for (int i = 0; i < tables.length; ++i) {
            StreamTableEnvironment tEnv =
                    (StreamTableEnvironment) ((TableImpl) tables[i]).getTableEnvironment();
            dataStreams.add(tEnv.toDataStream(tables[i]));
            nameToIndex.put(names[i], i);
        }
    }

    public MLData(DataStreamList dataStreamList) {
        this.dataStreams.addAll(dataStreamList.getDataStreams());
    }

    public void setCurrentProcessName(String name) {
        if (name != null && nameToIndex.containsKey(name)) {
            this.currentProcessId = nameToIndex.get(name);
        }
    }

    public static MLData of(Table... tables) {
        String[] names = new String[tables.length];
        for (int i = 0; i < names.length; ++i) {
            names[i] = "name_" + i;
        }
        return new MLData(tables, names);
    }

    public static MLData of(Table[] tables, String[] names) {
        return new MLData(tables, names);
    }

    public MLData merge(MLData data) {
        for (String name : data.nameToIndex.keySet()) {
            add(name, data.get(name));
        }
        return this;
    }

    public Table getTable() {
        StreamTableEnvironment tEnv =
                StreamTableEnvironment.create(
                        this.dataStreams.get(getCurrentProcessId()).getExecutionEnvironment());
        return tEnv.fromDataStream(this.dataStreams.get(getCurrentProcessId()));
    }

    public Table getTable(int i) {
        StreamTableEnvironment tEnv =
                StreamTableEnvironment.create(this.dataStreams.get(i).getExecutionEnvironment());
        return tEnv.fromDataStream(this.dataStreams.get(i));
    }

    public Table getTable(String name) {
        StreamTableEnvironment tEnv =
                StreamTableEnvironment.create(
                        this.dataStreams.get(nameToIndex.get(name)).getExecutionEnvironment());
        return tEnv.fromDataStream(this.dataStreams.get(nameToIndex.get(name)));
    }

    public Table[] getTables() {
        Table[] tables = new Table[dataStreams.size()];
        for (int i = 0; i < dataStreams.size(); ++i) {
            StreamTableEnvironment tEnv =
                    StreamTableEnvironment.create(
                            this.dataStreams.get(i).getExecutionEnvironment());
            tables[i] = tEnv.fromDataStream(this.dataStreams.get(i));
        }
        return tables;
    }

    public MLData slice(String... subNames) {
        DataStream<?>[] subData = new DataStream[subNames.length];
        for (int i = 0; i < subNames.length; ++i) {
            subData[i] = dataStreams.get(nameToIndex.get(subNames[i]));
        }
        return new MLData(subData, subNames);
    }

    public DataStream getCurrentDataStream() {
        return dataStreams.get(currentProcessId);
    }

    public List<DataStream<?>> getDataStreams() {
        return dataStreams;
    }

    public void add(String name, DataStream<?> dataStream) {
        if (name == null) {
            dataStreams.set(currentProcessId, dataStream);
        } else if (nameToIndex.containsKey(name)) {
            int id = nameToIndex.get(name);
            dataStreams.set(id, dataStream);
        } else {
            int id = dataStreams.size();
            dataStreams.add(id, dataStream);
            nameToIndex.put(name, id);
        }
    }

    public void add(String name, Table table) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) table).getTableEnvironment();
        if (name == null) {
            dataStreams.set(currentProcessId, tEnv.toDataStream(table));
        }
        if (nameToIndex.containsKey(name)) {
            int id = nameToIndex.get(name);
            dataStreams.set(id, tEnv.toDataStream(table));
        } else {
            int id = dataStreams.size();
            dataStreams.add(id, tEnv.toDataStream(table));
            nameToIndex.put(name, id);
        }
    }

    public void add(DataStream<?> dataStream) {
        dataStreams.add(dataStream);
    }

    public void add(Table table) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) table).getTableEnvironment();
        dataStreams.add(tEnv.toDataStream(table));
    }

    public int getParallelism() {
        return this.dataStreams.get(currentProcessId).getParallelism();
    }

    public StreamExecutionEnvironment getExecutionEnvironment() {
        return this.dataStreams.get(currentProcessId).getExecutionEnvironment();
    }

    public List<?> executeAndCollect(int i) throws Exception {
        return IteratorUtils.toList(dataStreams.get(i).executeAndCollect());
    }

    public ExecutionConfig getExecutionConfig() {
        return dataStreams.get(0).getExecutionConfig();
    }

    public int getCurrentProcessId() {
        return currentProcessId;
    }

    public DataStream get(int id) {
        return dataStreams.get(id);
    }

    public DataStream get(String name) {
        if (name == null) {
            return dataStreams.get(currentProcessId);
        } else if (nameToIndex.containsKey(name)) {
            return dataStreams.get(nameToIndex.get(name));
        } else {
            nameToIndex.put(name, nameToIndex.size());
            return dataStreams.get(nameToIndex.get(name));
        }
    }

    public MLData apply(CommonComponent component) {
        return component.apply(this);
    }

    @SuppressWarnings("unchecked")
    public void flatMap(
            String input,
            String output,
            String[] broadcastNames,
            Function function,
            TypeInformation<?> type,
            int parallel) {
        if (type == null) {
            type =
                    TypeExtractor.getFlatMapReturnTypes(
                            (FlatMapFunction<?, ?>) function, get(input).getType(), null, true);
        }
        DataStream dataStream = get(input);
        if (broadcastNames != null) {
            Map<String, DataStream<?>> broadcastMap = new HashMap<>();
            for (String name : broadcastNames) {
                broadcastMap.put(name, get(name));
            }
            final TypeInformation<?> finalType = type;
            dataStream =
                    BroadcastUtils.withBroadcastStream(
                            Collections.singletonList(dataStream),
                            broadcastMap,
                            inputList ->
                                    ((DataStream) inputList.get(0))
                                            .flatMap(
                                                    (RichFlatMapFunction<?, ?>) function,
                                                    finalType));
        } else {
            dataStream = dataStream.flatMap((FlatMapFunction<?, ?>) function, type);
        }
        if (parallel != -1) {
            dataStream.getTransformation().setParallelism(parallel);
        }
        add(output, dataStream);
        setCurrentProcessName(output);
    }

    public void flatMap(
            String input,
            String output,
            String[] broadcastNames,
            RichFlatMapFunction<?, ?> function,
            TypeInformation<?> type) {
        flatMap(input, output, broadcastNames, function, type, -1);
    }

    public void flatMap(String input, String output, FlatMapFunction<?, ?> function) {
        flatMap(input, output, null, function, null, -1);
    }

    public void flatMap(FlatMapFunction<?, ?> function) {
        flatMap(null, null, null, function, null, -1);
    }

    @SuppressWarnings("unchecked")
    public void mapPartition(
            String input,
            String output,
            String[] broadcastNames,
            Function function,
            TypeInformation<?> type,
            int parallel) {
        if (type == null) {
            type =
                    TypeExtractor.getMapPartitionReturnTypes(
                            (MapPartitionFunction<?, ?>) function,
                            get(input).getType(),
                            null,
                            true);
        }
        DataStream dataStream = get(input);
        if (broadcastNames != null) {
            Map<String, DataStream<?>> broadcastMap = new HashMap<>();
            for (String name : broadcastNames) {
                broadcastMap.put(name, get(name));
            }
            final TypeInformation<?> finalType = type;
            dataStream =
                    BroadcastUtils.withBroadcastStream(
                            Collections.singletonList(dataStream),
                            broadcastMap,
                            inputList ->
                                    DataStreamUtils.mapPartition(
                                            (DataStream) inputList.get(0),
                                            (RichMapPartitionFunction) function,
                                            finalType));

        } else {
            dataStream =
                    DataStreamUtils.mapPartition(dataStream, (MapPartitionFunction) function, type);
        }
        if (parallel != -1) {
            dataStream.getTransformation().setParallelism(parallel);
        }
        add(output, dataStream);
        setCurrentProcessName(output);
    }

    public void mapPartition(
            String input,
            String output,
            String[] broadcastNames,
            RichMapFunction<?, ?> function,
            TypeInformation<?> type) {
        mapPartition(input, output, broadcastNames, function, type, -1);
    }

    public void mapPartition(String input, String output, MapFunction<?, ?> function) {
        mapPartition(input, output, null, function, null, -1);
    }

    public void mapPartition(MapPartitionFunction<?, ?> function) {
        mapPartition(null, null, null, function, null, -1);
    }

    @SuppressWarnings("unchecked")
    public void map(
            String input,
            String output,
            String[] broadcastNames,
            Function function,
            TypeInformation<?> type,
            int parallel) {
        if (type == null) {
            type =
                    TypeExtractor.getMapReturnTypes(
                            (MapFunction<?, ?>) function, get(input).getType(), null, true);
        }
        DataStream dataStream = get(input);
        if (broadcastNames != null) {
            Map<String, DataStream<?>> broadcastMap = new HashMap<>();
            for (String name : broadcastNames) {
                broadcastMap.put(name, get(name));
            }
            final TypeInformation<?> finalType = type;
            dataStream =
                    BroadcastUtils.withBroadcastStream(
                            Collections.singletonList(dataStream),
                            broadcastMap,
                            inputList ->
                                    ((DataStream) inputList.get(0))
                                            .map((RichMapFunction<?, ?>) function, finalType));
        } else {
            dataStream = dataStream.map((MapFunction<?, ?>) function, type);
        }
        if (parallel != -1) {
            dataStream.getTransformation().setParallelism(parallel);
        }
        add(output, dataStream);
        setCurrentProcessName(output);
    }

    public void map(
            String input,
            String output,
            String[] broadcastNames,
            MapFunction<?, ?> function,
            TypeInformation<?> type) {
        map(input, output, broadcastNames, function, type, -1);
    }

    public void map(String input, String output, MapFunction<?, ?> function) {
        map(input, output, null, function, null, -1);
    }

    public void map(MapFunction<?, ?> function, TypeInformation<?> type) {
        map(null, null, null, function, type, -1);
    }

    public void map(MapFunction<?, ?> function) {
        map(null, null, null, function, null, -1);
    }

    @SuppressWarnings("unchecked")
    public void keyBy(String input, String output, Function function) {
        DataStream dataStream = get(input);

        dataStream = dataStream.keyBy((KeySelector<?, ?>) function);
        add(output, dataStream);

        setCurrentProcessName(output);
    }

    public void keyBy(Function function) {
        keyBy(null, null, function);
    }

    @SuppressWarnings("unchecked")
    public void reduce(String input, String output, Function function, boolean isOnline) {
        DataStream<?> dataStream;
        if (isOnline) {
            dataStream = get(input);
            dataStream =
                    dataStream
                            .countWindowAll(dataStream.getParallelism())
                            .reduce((ReduceFunction) function);
        } else {
            dataStream = DataStreamUtils.reduce(get(input), (ReduceFunction<?>) function);
        }
        add(output, dataStream);
        setCurrentProcessName(output);
    }

    public void reduce(String input, String output, Function function) {
        reduce(input, output, function, false);
    }

    public void reduce(Function function, boolean isOnline) {
        reduce(null, null, function, isOnline);
    }

    public void reduce(Function function) {
        reduce(null, null, function, false);
    }

    public void shuffle(String input, String output) {
        add(output, get(input).shuffle());
        setCurrentProcessName(output);
    }

    public void rebalance(String input, String output) {
        add(output, get(input).rebalance());
        setCurrentProcessName(output);
    }

    @SuppressWarnings("unchecked")
    public void union(String input, String output, String withName) {
        add(output, get(input).union(get(withName)));
        setCurrentProcessName(output);
    }

    public void union(String withName) {
        union(null, null, withName);
    }

    @SuppressWarnings("unchecked")
    public void allReduce(String inputName, String outputName, boolean isForEachRound) {
        if (isForEachRound) {
            DataStream<?> output =
                    IterationBody.forEachRound(
                                    DataStreamList.of(get(inputName)),
                                    input -> {
                                        DataStream<double[]> feedback =
                                                DataStreamUtils.allReduceSum(input.get(0));
                                        return DataStreamList.of(feedback);
                                    })
                            .get(0);
            add(outputName, output);

        } else {
            add(outputName, DataStreamUtils.allReduceSum(get(inputName)));
        }
        setCurrentProcessName(outputName);
    }

    @SuppressWarnings("unchecked")
    public void aggregate(String input, String output, Function function, int parallel) {

        DataStream<?> dataStream =
                DataStreamUtils.aggregate(get(input), (AggregateFunction<?, ?, ?>) function);

        if (parallel != -1) {
            dataStream.getTransformation().setParallelism(parallel);
        }
        add(output, dataStream);
        setCurrentProcessName(output);
    }

    @SuppressWarnings("unchecked")
    public void aggregate(String input, String output, Function function) {
        aggregate(input, output, function, -1);
    }

    public void aggregate(Function function) {
        aggregate(null, null, function, -1);
    }

    public void broadcast(String input, String output) {
        add(output, get(input).broadcast());
        setCurrentProcessName(output);
    }

    public void groupReduce(String input, String output, final Function function) {

        DataStream<?> dataStream = get(input);

        dataStream =
                ((KeyedStream) dataStream)
                        .window(EndOfStreamWindows.get())
                        .process((ProcessWindowFunction) function);
        add(output, dataStream);
        setCurrentProcessName(output);
    }

    /** Comments. */
    public static class MLDataFunction implements CommonComponent {
        String functionName;
        String input;
        String output;
        String withName;
        String[] broadcastNames;
        TypeInformation<?> type;
        int parallel = -1;
        Function function;
        boolean isForEachRound = false;
        boolean isOnline = false;

        public MLDataFunction(String functionName, Function function) {
            this.functionName = functionName;
            this.function = function;
        }

        public MLDataFunction(String functionName) {
            this.functionName = functionName;
        }

        public final MLDataFunction input(String fromName) {
            this.input = fromName;
            return this;
        }

        public final MLDataFunction output(String toName) {
            this.output = toName;
            return this;
        }

        public final MLDataFunction returns(TypeInformation type) {
            this.type = type;
            return this;
        }

        public final MLDataFunction withParallel(int parallel) {
            this.parallel = parallel;
            return this;
        }

        public MLDataFunction with(String withName) {
            this.withName = withName;
            return this;
        }

        public final MLDataFunction withBroadcast(String... name) {
            this.broadcastNames = name;
            return this;
        }

        public final MLDataFunction isForEachRound(boolean isForEachRound) {
            this.isForEachRound = isForEachRound;
            return this;
        }

        public final MLDataFunction isOnine(boolean isOnline) {
            this.isOnline = isOnline;
            return this;
        }

        @Override
        public MLData apply(MLData mlData) {
            switch (FUNCTION.valueOf(functionName.toUpperCase())) {
                case FLATMAP:
                    mlData.flatMap(input, output, broadcastNames, function, type, parallel);
                    break;
                case MAP:
                    mlData.map(input, output, broadcastNames, function, type, parallel);
                    break;
                case MAPPARTITION:
                    mlData.mapPartition(input, output, broadcastNames, function, type, parallel);
                    break;
                case AGGREGATE:
                    mlData.aggregate(input, output, function, parallel);
                    break;
                case BROADCAST:
                    mlData.broadcast(input, output);
                    break;
                case ALLREDUCE:
                    mlData.allReduce(input, output, isForEachRound);
                    break;
                case KEYBY:
                    mlData.keyBy(input, output, function);
                    break;
                case REDUCE:
                    mlData.reduce(input, output, function, isOnline);
                    break;
                case SHUFFLE:
                    mlData.shuffle(input, output);
                    break;
                case REBALANCE:
                    mlData.rebalance(input, output);
                    break;
                case UNION:
                    mlData.union(input, output, withName);
                    break;
                case GROUPREDUCE:
                    mlData.groupReduce(input, output, function);
                    break;
                default:
            }
            return mlData;
        }
    }

    enum FUNCTION {
        FLATMAP,
        MAP,
        MAPPARTITION,
        FILTER,
        AGGREGATE,
        COGROUP,
        COFLATMAP,
        REDUCE,
        JOIN,
        BROADCAST,
        ALLREDUCE,
        KEYBY,
        SHUFFLE,
        REBALANCE,
        UNION,
        GROUPREDUCE
    }
}
