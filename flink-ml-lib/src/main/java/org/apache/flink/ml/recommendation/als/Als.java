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

package org.apache.flink.ml.recommendation.als;

import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.ps.api.AlgorithmFlow;
import org.apache.flink.ml.common.ps.api.MLData;
import org.apache.flink.ml.common.ps.api.MLData.MLDataFunction;
import org.apache.flink.ml.common.ps.iterations.PsAllReduceComponent;
import org.apache.flink.ml.common.ps.iterations.PullComponent;
import org.apache.flink.ml.common.ps.iterations.PushComponent;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;
import org.apache.flink.util.function.SerializableFunction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An Estimator which implements the Als algorithm.
 *
 * <p>ALS tries to decompose a matrix R as R = X * Yt. Here X and Y are called factor matrices.
 * Matrix R is usually a sparse matrix representing ratings given from users to items. ALS tries to
 * find X and Y that minimize || R - X * Yt ||^2. This is done by iterations. At each step, X is
 * fixed and Y is solved, then Y is fixed and X is solved.
 *
 * <p>The algorithm is described in "Large-scale Parallel Collaborative Filtering for the Netflix
 * Prize, 2007". This algorithm also supports implicit preference model described in "Collaborative
 * Filtering for Implicit Feedback Datasets, 2008".
 */
public class Als implements Estimator<Als, AlsModel>, AlsParams<Als> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private static final int THRESHOLD = 100000;

    public Als() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public AlsModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        MLData mlData = MLData.of(inputs, new String[] {"data"});
        AlsMLSession mlSession =
                new AlsMLSession(getImplicitPrefs(), getRank(), mlData.getParallelism());
        TypeSerializer<double[]> typeSerializer =
                TypeInformation.of(double[].class).createSerializer(mlData.getExecutionConfig());
        AlsModelUpdater updater = new AlsModelUpdater(getRank());

        AlgorithmFlow algorithmFlow = new AlgorithmFlow();

        if (getImplicitPrefs()) {
            /*
             * If using implicit prefs, the whole yty matrix must be computed by all reduce stage.
             */
            algorithmFlow
                    .add(
                            new MLDataFunction(
                                            "map",
                                            new TransformSample(
                                                    getUserCol(), getItemCol(), getRatingCol()))
                                    .returns(
                                            new TupleTypeInfo<>(
                                                    Types.LONG, Types.LONG, Types.DOUBLE)))
                    .add(
                            new MLDataFunction("flatMap", new DuplicateSample())
                                    .returns(
                                            new TupleTypeInfo<>(
                                                    Types.LONG, Types.LONG, Types.DOUBLE)))
                    .add(
                            new MLDataFunction(
                                    "keyBy",
                                    (KeySelector<Tuple3<Long, Long, Double>, Long>)
                                            value -> value.f0))
                    .add(new MLDataFunction("groupReduce", new GenerateRatings(THRESHOLD)))
                    .add(new MLDataFunction("rebalance"))
                    .add(
                            new MLDataFunction("mapPartition", new CalcLocalProfile())
                                    .output("profile"))
                    .add(new MLDataFunction("reduce", new ReduceProfile()))
                    .add(new MLDataFunction("map", new GenerateFinalProfile()))
                    .add(new MLDataFunction("broadcast"))
                    .add(new MLDataFunction("union").with("data").output("data"))
                    .startServerIteration(mlSession, updater)
                    .add(new ComputeYtyIndices())
                    .add(
                            new PullComponent(
                                    () -> mlSession.pullIndices,
                                    () -> mlSession.aggregatorSDAArray,
                                    new YtyAggregator()))
                    .add(new CopyAllReduceData(getRank()))
                    .add(
                            new PsAllReduceComponent<>(
                                    () -> mlSession.allReduceBuffer,
                                    () -> mlSession.allReduceBuffer,
                                    (ReduceFunction<double[][]>) Als::sumYty,
                                    typeSerializer,
                                    1))
                    .add(new ComputeNeighborIndices(getRank()))
                    .add(new PullComponent(() -> mlSession.pullIndices, () -> mlSession.pullValues))
                    .add(
                            new UpdateCommonFactors(
                                    getRank(),
                                    getImplicitPrefs(),
                                    getNonNegative(),
                                    getRegParam(),
                                    getAlpha()))
                    .add(new PushComponent(() -> mlSession.pushIndices, () -> mlSession.pushValues))
                    .add(
                            new ComputeLsMatrixVector(
                                    getRank(), getImplicitPrefs(), getRegParam(), getAlpha()))
                    .add(new PushComponent(() -> mlSession.pushIndices, () -> mlSession.pushValues))
                    .add(new PullComponent(() -> mlSession.pullIndices, () -> mlSession.pullValues))
                    .add(new UpdateHotPointFactors(getRank(), getNonNegative()))
                    .add(new PushComponent(() -> mlSession.pushIndices, () -> mlSession.pushValues))
                    .endServerIteration(
                            (SerializableFunction<AlsMLSession, Boolean>)
                                    o ->
                                            o.iterationId / (o.numItemBlocks + o.numUserBlocks)
                                                    >= getMaxIter())
                    .add(new MLDataFunction("mapPartition", new Als.GenerateModelData()));
        } else {
            algorithmFlow
                    .add(
                            new MLDataFunction(
                                            "map",
                                            new TransformSample(
                                                    getUserCol(), getItemCol(), getRatingCol()))
                                    .returns(
                                            new TupleTypeInfo<>(
                                                    Types.LONG, Types.LONG, Types.DOUBLE)))
                    .add(
                            new MLDataFunction("flatMap", new DuplicateSample())
                                    .returns(
                                            new TupleTypeInfo<>(
                                                    Types.LONG, Types.LONG, Types.DOUBLE)))
                    .add(
                            new MLDataFunction(
                                    "keyBy",
                                    (KeySelector<Tuple3<Long, Long, Double>, Long>)
                                            value -> value.f0))
                    .add(new MLDataFunction("groupReduce", new GenerateRatings(THRESHOLD)))
                    .add(new MLDataFunction("rebalance"))
                    .add(
                            new MLDataFunction("mapPartition", new CalcLocalProfile())
                                    .output("profile"))
                    .add(new MLDataFunction("reduce", new ReduceProfile()))
                    .add(new MLDataFunction("map", new GenerateFinalProfile()))
                    .add(new MLDataFunction("broadcast"))
                    .add(new MLDataFunction("union").with("data").output("data"))
                    .startServerIteration(mlSession, updater)
                    .add(new ComputeNeighborIndices(getRank()))
                    .add(new PullComponent(() -> mlSession.pullIndices, () -> mlSession.pullValues))
                    .add(
                            new UpdateCommonFactors(
                                    getRank(),
                                    getImplicitPrefs(),
                                    getNonNegative(),
                                    getRegParam(),
                                    getAlpha()))
                    .add(new PushComponent(() -> mlSession.pushIndices, () -> mlSession.pushValues))
                    .add(
                            new ComputeLsMatrixVector(
                                    getRank(), getImplicitPrefs(), getRegParam(), getAlpha()))
                    .add(new PushComponent(() -> mlSession.pushIndices, () -> mlSession.pushValues))
                    .add(new PullComponent(() -> mlSession.pullIndices, () -> mlSession.pullValues))
                    .add(new UpdateHotPointFactors(getRank(), getNonNegative()))
                    .add(new PushComponent(() -> mlSession.pushIndices, () -> mlSession.pushValues))
                    .endServerIteration(
                            (SerializableFunction<AlsMLSession, Boolean>)
                                    o ->
                                            o.iterationId / (o.numItemBlocks + o.numUserBlocks)
                                                    >= getMaxIter())
                    .add(new MLDataFunction("mapPartition", new Als.GenerateModelData()));
        }

        AlsModel model = new AlsModel().setModelData(algorithmFlow.apply(mlData).getTables());
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    /** Generates the ModelData from the results of iteration. */
    private static class GenerateModelData implements MapPartitionFunction<Object, AlsModelData> {

        private final List<Tuple2<Long, float[]>> userFactors = new ArrayList<>();
        private final List<Tuple2<Long, float[]>> itemFactors = new ArrayList<>();

        @Override
        @SuppressWarnings("unchecked")
        public void mapPartition(Iterable<Object> iterable, Collector<AlsModelData> collector) {
            for (Object ele : iterable) {
                Tuple2<Long, float[]> t2 = (Tuple2<Long, float[]>) ele;

                if (t2.f0 % 2L == 1L) {
                    long id = (t2.f0 - 1) / 2L;
                    itemFactors.add(Tuple2.of(id, t2.f1));
                } else {
                    long id = t2.f0 / 2L;
                    userFactors.add(Tuple2.of(id, t2.f1));
                }
            }
            collector.collect(new AlsModelData(userFactors, itemFactors));
        }
    }

    private static double[][] sumYty(double[][] d1, double[][] d2) {
        Preconditions.checkArgument(d1[0].length == d2[0].length);
        for (int i = 0; i < d1[0].length; i++) {
            d2[0][i] += d1[0][i];
        }
        return d2;
    }

    /** The whole ratings of a user or an item. */
    public static class Ratings {

        public Ratings() {}

        /** Current node is a split node or not. */
        public boolean isSplit;

        /** Current node is a main node in split nodes or not. */
        public boolean isMainNode;

        /** UserId or itemId decided by identity. */
        public long nodeId;

        /** Number of neighbors. */
        public int numNeighbors;

        /** Neighbors of this nodeId. */
        public long[] neighbors;

        /** Scores from neighbors to this nodeId. */
        public double[] scores;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static Als load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
