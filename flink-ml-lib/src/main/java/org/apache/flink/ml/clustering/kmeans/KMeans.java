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

package org.apache.flink.ml.clustering.kmeans;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.ObjectArrayTypeInfo;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.distance.DistanceMeasure;
import org.apache.flink.ml.common.ps.api.AlgorithmFlow;
import org.apache.flink.ml.common.ps.api.CoTransformComponent;
import org.apache.flink.ml.common.ps.api.MLData;
import org.apache.flink.ml.common.ps.api.MLData.MLDataFunction;
import org.apache.flink.ml.common.ps.api.ManagedMemoryComponent;
import org.apache.flink.ml.common.ps.api.SampleComponent;
import org.apache.flink.ml.common.ps.api.TerminateComponent;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.VectorWithNorm;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.VectorWithNormSerializer;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * An Estimator which implements the k-means clustering algorithm.
 *
 * <p>See https://en.wikipedia.org/wiki/K-means_clustering.
 */
public class KMeans implements Estimator<KMeans, KMeansModel>, KMeansParams<KMeans> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public KMeans() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public KMeansModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        MLData mlData = MLData.of(inputs, new String[] {"data"});
        AlgorithmFlow algorithmFlow = new AlgorithmFlow();
        algorithmFlow
                .add(
                        new MLDataFunction(
                                "map",
                                (MapFunction<Row, DenseVector>)
                                        row -> ((Vector) row.getField(getFeaturesCol())).toDense()))
                .add(new SampleComponent(getK(), getSeed()).output("initCentroids"))
                .add(
                        new MLDataFunction(
                                        "mapPartition",
                                        (MapPartitionFunction<DenseVector, DenseVector[]>)
                                                (iterable, collector) -> {
                                                    List<DenseVector> list = new ArrayList<>();
                                                    iterable.iterator().forEachRemaining(list::add);
                                                    collector.collect(
                                                            list.toArray(new DenseVector[0]));
                                                })
                                .returns(TypeInformation.of(DenseVector[].class))
                                .withParallel(1))
                .startIteration(new String[] {"initCentroids"}, new String[] {"data"}, false)
                .add(
                        new TerminateOnMaxComponent<DenseVector[]>(getMaxIter())
                                .output("terminateData"))
                .add(new MLDataFunction("broadcast").input("initCentroids").output("initCentroids"))
                .add(
                        new CentroidsUpdateAccumulator(
                                        DistanceMeasure.getInstance(getDistanceMeasure()))
                                .input("data")
                                .with("initCentroids")
                                .output("centroidIdAndPoints")
                                .withOutType(
                                        new TupleTypeInfo<>(
                                                BasicArrayTypeInfo.INT_ARRAY_TYPE_INFO,
                                                ObjectArrayTypeInfo.getInfoFor(
                                                        DenseVectorTypeInfo.INSTANCE))))
                .add(new ManagedMemoryComponent(100).input("centroidIdAndPoints"))
                .add(
                        new MLDataFunction("reduce", new CentroidsUpdateReducer())
                                .isOnine(true)
                                .output("newModelData"))
                .add(new MLDataFunction("map", new ModelDataGenerator()))
                .add(
                        new MLDataFunction(
                                        "map",
                                        (MapFunction<KMeansModelData, DenseVector[]>)
                                                kMeansModelData -> kMeansModelData.centroids)
                                .withParallel(1)
                                .output("newCentroids"))
                .add(
                        new MLDataFunction("flatMap", new ForwardInputsOfLastRound())
                                .input("newModelData")
                                .output("finalModel")
                                .returns(TypeInformation.of(KMeansModelData.class)))
                .endIteration(new String[] {"finalModel", "newCentroids", "terminateData"}, false);

        Table finalModelDataTable = algorithmFlow.apply(mlData).getTable();
        KMeansModel model = new KMeansModel().setModelData(finalModelDataTable);
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    /** Comments. */
    public static class TerminateOnMaxComponent<T> extends TerminateComponent<T, Integer> {

        private final int maxIter;

        private double loss = Double.MAX_VALUE;

        public TerminateOnMaxComponent(Integer maxIter) {
            this.maxIter = maxIter;
        }

        @Override
        public void flatMap(T value, Collector<Integer> out) {}

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<Integer> collector) {
            if ((epochWatermark + 1) < maxIter) {
                collector.collect(0);
            }
            loss = Double.MAX_VALUE;
        }

        @Override
        public void onIterationTerminated(Context context, Collector<Integer> collector) {}
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static KMeans load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /** Comments. */
    private static class CentroidsUpdateReducer
            implements ReduceFunction<Tuple2<Integer[], DenseVector[]>> {
        @Override
        public Tuple2<Integer[], DenseVector[]> reduce(
                Tuple2<Integer[], DenseVector[]> tuple2, Tuple2<Integer[], DenseVector[]> t1)
                throws Exception {
            for (int i = 0; i < tuple2.f0.length; i++) {
                tuple2.f0[i] += t1.f0[i];
                BLAS.axpy(1.0, t1.f1[i], tuple2.f1[i]);
            }

            return tuple2;
        }
    }

    /** Comments. */
    private static class ModelDataGenerator
            implements MapFunction<Tuple2<Integer[], DenseVector[]>, KMeansModelData> {
        @Override
        public KMeansModelData map(Tuple2<Integer[], DenseVector[]> tuple2) throws Exception {
            double[] weights = new double[tuple2.f0.length];
            for (int i = 0; i < tuple2.f0.length; i++) {
                BLAS.scal(1.0 / tuple2.f0[i], tuple2.f1[i]);
                weights[i] = tuple2.f0[i];
            }

            return new KMeansModelData(tuple2.f1, new DenseVector(weights));
        }
    }

    private static class CentroidsUpdateAccumulator
            extends CoTransformComponent<
                    DenseVector, DenseVector[], Tuple2<Integer[], DenseVector[]>> {

        private final DistanceMeasure distanceMeasure;

        private ListState<DenseVector[]> centroids;

        private ListStateWithCache<VectorWithNorm> points;

        public CentroidsUpdateAccumulator(DistanceMeasure distanceMeasure) {
            super();
            this.distanceMeasure = distanceMeasure;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);

            TypeInformation<DenseVector[]> type =
                    ObjectArrayTypeInfo.getInfoFor(DenseVectorTypeInfo.INSTANCE);

            centroids =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("centroids", type));

            points =
                    new ListStateWithCache<>(
                            new VectorWithNormSerializer(),
                            getContainingTask(),
                            getRuntimeContext(),
                            context,
                            config.getOperatorID());
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            points.snapshotState(context);
        }

        @Override
        public void processElement1(StreamRecord<DenseVector> streamRecord) throws Exception {
            points.add(new VectorWithNorm(streamRecord.getValue()));
        }

        @Override
        public void processElement2(StreamRecord<DenseVector[]> streamRecord) throws Exception {
            Preconditions.checkState(!centroids.get().iterator().hasNext());
            centroids.add(streamRecord.getValue());
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark,
                Context context,
                Collector<Tuple2<Integer[], DenseVector[]>> out)
                throws Exception {
            DenseVector[] centroidValues =
                    Objects.requireNonNull(
                            OperatorStateUtils.getUniqueElement(centroids, "centroids")
                                    .orElse(null));

            VectorWithNorm[] centroidsWithNorm = new VectorWithNorm[centroidValues.length];
            for (int i = 0; i < centroidsWithNorm.length; i++) {
                centroidsWithNorm[i] = new VectorWithNorm(centroidValues[i]);
            }

            DenseVector[] newCentroids = new DenseVector[centroidValues.length];
            Integer[] counts = new Integer[centroidValues.length];
            Arrays.fill(counts, 0);
            for (int i = 0; i < centroidValues.length; i++) {
                newCentroids[i] = new DenseVector(centroidValues[i].size());
            }

            for (VectorWithNorm point : points.get()) {
                int closestCentroidId = distanceMeasure.findClosest(centroidsWithNorm, point);
                BLAS.axpy(1.0, point.vector, newCentroids[closestCentroidId]);
                counts[closestCentroidId]++;
            }

            output.collect(new StreamRecord<>(Tuple2.of(counts, newCentroids)));
            centroids.clear();
        }

        @Override
        public void onIterationTerminated(
                Context context, Collector<Tuple2<Integer[], DenseVector[]>> collector) {
            centroids.clear();
            points.clear();
        }
    }

    /** Comments. */
    public static class ForwardInputsOfLastRound<T> extends RichFlatMapFunction<T, T>
            implements IterationListener<T> {
        private List<T> valuesInLastEpoch = new ArrayList<>();
        private List<T> valuesInCurrentEpoch = new ArrayList<>();

        @Override
        public void flatMap(T value, Collector<T> out) {
            valuesInCurrentEpoch.add(value);
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<T> out) {
            valuesInLastEpoch = valuesInCurrentEpoch;
            valuesInCurrentEpoch = new ArrayList<>();
        }

        @Override
        public void onIterationTerminated(Context context, Collector<T> out) {
            for (T value : valuesInLastEpoch) {
                out.collect(value);
            }
            if (!valuesInCurrentEpoch.isEmpty()) {
                throw new IllegalStateException(
                        "flatMap() is invoked since the last onEpochWatermarkIncremented callback");
            }
            valuesInLastEpoch.clear();
        }
    }
}
