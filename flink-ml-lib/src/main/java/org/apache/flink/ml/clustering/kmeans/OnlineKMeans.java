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
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.ObjectArrayTypeInfo;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.distance.DistanceMeasure;
import org.apache.flink.ml.common.ps.api.AlgorithmFlow;
import org.apache.flink.ml.common.ps.api.CoTransformComponent;
import org.apache.flink.ml.common.ps.api.MLData;
import org.apache.flink.ml.common.ps.api.MLData.MLDataFunction;
import org.apache.flink.ml.common.ps.api.MiniBatchComponent;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.VectorWithNorm;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * OnlineKMeans extends the function of {@link KMeans}, supporting to train a K-Means model
 * continuously according to an unbounded stream of train data.
 *
 * <p>OnlineKMeans makes updates with the "mini-batch" KMeans rule, generalized to incorporate
 * forgetfulness (i.e. decay). After the centroids estimated on the current batch are acquired,
 * OnlineKMeans computes the new centroids from the weighted average between the original and the
 * estimated centroids. The weight of the estimated centroids is the number of points assigned to
 * them. The weight of the original centroids is also the number of points, but additionally
 * multiplying with the decay factor.
 *
 * <p>The decay factor scales the contribution of the clusters as estimated thus far. If the decay
 * factor is 1, all batches are weighted equally. If the decay factor is 0, new centroids are
 * determined entirely by recent data. Lower values correspond to more forgetting.
 */
public class OnlineKMeans
        implements Estimator<OnlineKMeans, OnlineKMeansModel>, OnlineKMeansParams<OnlineKMeans> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table initModelDataTable;

    public OnlineKMeans() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public OnlineKMeansModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        MLData mlData =
                MLData.of(
                        new Table[] {inputs[0], initModelDataTable},
                        new String[] {"data", "model"});

        AlgorithmFlow algorithmFlow =
                new AlgorithmFlow(false)
                        .add(new MLDataFunction("map", new FeaturesExtractor(getFeaturesCol())))
                        .add(
                                new MLDataFunction(
                                                "map",
                                                (MapFunction<Row, KMeansModelData>)
                                                        (x ->
                                                                new KMeansModelData(
                                                                        Arrays.stream(
                                                                                        ((Vector[])
                                                                                                x
                                                                                                        .getField(
                                                                                                                0)))
                                                                                .map(
                                                                                        Vector
                                                                                                ::toDense)
                                                                                .toArray(
                                                                                        DenseVector
                                                                                                        []
                                                                                                ::new),
                                                                        ((Vector) x.getField(1))
                                                                                .toDense())))
                                        .withParallel(1)
                                        .input("model")
                                        .output("model"))
                        .startIteration(new String[] {"model"}, new String[] {"data"}, false)
                        .add(
                                new MiniBatchComponent(getGlobalBatchSize())
                                        .input("data")
                                        .output("data"))
                        .add(new MLDataFunction("broadcast").input("model").output("model"))
                        .add(
                                new ModelDataLocalUpdater(
                                                DistanceMeasure.getInstance(getDistanceMeasure()),
                                                getK(),
                                                getDecayFactor())
                                        .input("data")
                                        .with("model")
                                        .output("newModel")
                                        .returns(TypeInformation.of(KMeansModelData.class)))
                        .add(
                                new MLDataFunction("reduce", new ModelDataGlobalReducer())
                                        .isOnine(true))
                        .endIteration(new String[] {"newModel", "model"}, false);

        Table onlineModelDataTable = algorithmFlow.apply(mlData).getTable();
        OnlineKMeansModel model = new OnlineKMeansModel().setModelData(onlineModelDataTable);
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    /** Saves the metadata AND bounded model data table (if exists) to the given path. */
    @Override
    public void save(String path) throws IOException {
        Preconditions.checkNotNull(
                initModelDataTable, "Initial Model Data Table should have been set.");
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                KMeansModelDataUtil.getModelDataStream(initModelDataTable),
                path,
                new KMeansModelDataUtil.ModelDataEncoder());
    }

    public static OnlineKMeans load(StreamTableEnvironment tEnv, String path) throws IOException {
        OnlineKMeans onlineKMeans = ReadWriteUtils.loadStageParam(path);
        onlineKMeans.initModelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new KMeansModelDataUtil.ModelDataDecoder());
        return onlineKMeans;
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * Operator that collects a KMeansModelData from each upstream subtask, and outputs the weight
     * average of collected model data.
     */
    private static class ModelDataGlobalReducer implements ReduceFunction<KMeansModelData> {
        @Override
        public KMeansModelData reduce(KMeansModelData modelData, KMeansModelData newModelData) {
            DenseVector weights = modelData.weights;
            DenseVector[] centroids = modelData.centroids;
            DenseVector newWeights = newModelData.weights;
            DenseVector[] newCentroids = newModelData.centroids;

            int k = newCentroids.length;
            int dim = newCentroids[0].size();

            for (int i = 0; i < k; i++) {
                for (int j = 0; j < dim; j++) {
                    centroids[i].values[j] =
                            (centroids[i].values[j] * weights.values[i]
                                            + newCentroids[i].values[j] * newWeights.values[i])
                                    / Math.max(weights.values[i] + newWeights.values[i], 1e-16);
                }
                weights.values[i] += newWeights.values[i];
            }

            return new KMeansModelData(centroids, weights);
        }
    }

    /**
     * An operator that updates KMeans model data locally. It mainly does the following operations.
     *
     * <ul>
     *   <li>Finds the closest centroid id (cluster) of the input points.
     *   <li>Computes the new centroids from the average of input points that belongs to the same
     *       cluster.
     *   <li>Computes the weighted average of current and new centroids. The weight of a new
     *       centroid is the number of input points that belong to this cluster. The weight of a
     *       current centroid is its original weight scaled by $ decayFactor / parallelism $.
     *   <li>Generates new model data from the weighted average of centroids, and the sum of
     *       weights.
     * </ul>
     */
    private static class ModelDataLocalUpdater
            extends CoTransformComponent<DenseVector[], KMeansModelData, KMeansModelData> {
        private final DistanceMeasure distanceMeasure;
        private final int k;
        private final double decayFactor;
        private ListState<DenseVector[]> localBatchDataState;
        private ListState<KMeansModelData> modelDataState;

        private ModelDataLocalUpdater(DistanceMeasure distanceMeasure, int k, double decayFactor) {
            this.distanceMeasure = distanceMeasure;
            this.k = k;
            this.decayFactor = decayFactor;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);

            TypeInformation<DenseVector[]> type =
                    ObjectArrayTypeInfo.getInfoFor(DenseVectorTypeInfo.INSTANCE);
            localBatchDataState =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("localBatch", type));

            modelDataState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>("modelData", KMeansModelData.class));
        }

        @Override
        public void processElement1(StreamRecord<DenseVector[]> pointsRecord) throws Exception {
            localBatchDataState.add(pointsRecord.getValue());
            alignAndComputeModelData();
        }

        @Override
        public void processElement2(StreamRecord<KMeansModelData> modelDataRecord)
                throws Exception {
            Preconditions.checkArgument(modelDataRecord.getValue().centroids.length == k);

            modelDataState.add(modelDataRecord.getValue());
            alignAndComputeModelData();
        }

        private void alignAndComputeModelData() throws Exception {
            if (!modelDataState.get().iterator().hasNext()
                    || !localBatchDataState.get().iterator().hasNext()) {
                return;
            }

            KMeansModelData modelData =
                    OperatorStateUtils.getUniqueElement(modelDataState, "modelData").get();
            DenseVector[] centroids = modelData.centroids;
            VectorWithNorm[] centroidsWithNorm = new VectorWithNorm[modelData.centroids.length];
            for (int i = 0; i < centroidsWithNorm.length; i++) {
                centroidsWithNorm[i] = new VectorWithNorm(modelData.centroids[i]);
            }
            DenseVector weights = modelData.weights;
            modelDataState.clear();

            List<DenseVector[]> pointsList =
                    IteratorUtils.toList(localBatchDataState.get().iterator());
            DenseVector[] points = pointsList.remove(0);
            localBatchDataState.update(pointsList);

            int dim = centroids[0].size();
            int parallelism = getRuntimeContext().getNumberOfParallelSubtasks();

            // Computes new centroids.
            DenseVector[] sums = new DenseVector[k];
            int[] counts = new int[k];

            for (int i = 0; i < k; i++) {
                sums[i] = new DenseVector(dim);
            }
            for (DenseVector point : points) {
                int closestCentroidId =
                        distanceMeasure.findClosest(centroidsWithNorm, new VectorWithNorm(point));
                counts[closestCentroidId]++;
                BLAS.axpy(1.0, point, sums[closestCentroidId]);
            }

            // Considers weight and decay factor when updating centroids.
            BLAS.scal(decayFactor / parallelism, weights);
            for (int i = 0; i < k; i++) {
                if (counts[i] == 0) {
                    continue;
                }

                DenseVector centroid = centroids[i];
                weights.values[i] = weights.values[i] + counts[i];
                double lambda = counts[i] / weights.values[i];

                BLAS.scal(1.0 - lambda, centroid);
                BLAS.axpy(lambda / counts[i], sums[i], centroid);
            }

            output.collect(new StreamRecord<>(new KMeansModelData(centroids, weights)));
        }
    }

    private static class FeaturesExtractor implements MapFunction<Row, DenseVector> {
        private final String featuresCol;

        private FeaturesExtractor(String featuresCol) {
            this.featuresCol = featuresCol;
        }

        @Override
        public DenseVector map(Row row) {
            return ((Vector) row.getField(featuresCol)).toDense();
        }
    }

    /**
     * Sets the initial model data of the online training process with the provided model data
     * table.
     */
    public OnlineKMeans setInitialModelData(Table initModelDataTable) {
        this.initModelDataTable = initModelDataTable;
        return this;
    }
}
