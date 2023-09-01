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

package org.apache.flink.ml.common.gbt.operators;

import org.apache.flink.api.java.tuple.Tuple5;
import org.apache.flink.ml.common.gbt.DataUtils;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.Slice;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.function.Consumer;
import java.util.stream.IntStream;

import static org.apache.flink.ml.common.gbt.DataUtils.BIN_SIZE;

class HistBuilder {
    private static final Logger LOG = LoggerFactory.getLogger(HistBuilder.class);

    private final int subtaskId;

    private final int numFeatures;
    private final int[] numFeatureBins;
    private final FeatureMeta[] featureMetas;

    private final int numBaggingFeatures;
    private final Random featureRandomizer;
    private final int[] featureIndicesPool;

    private final boolean isUnseenMissing;
    private final int maxDepth;

    public HistBuilder(TrainContext trainContext) {
        subtaskId = trainContext.subtaskId;

        numFeatures = trainContext.numFeatures;
        numFeatureBins = trainContext.numFeatureBins;
        featureMetas = trainContext.featureMetas;

        numBaggingFeatures = trainContext.numBaggingFeatures;
        featureRandomizer = trainContext.featureRandomizer;
        featureIndicesPool = IntStream.range(0, trainContext.numFeatures).toArray();

        isUnseenMissing = trainContext.isUnseenMissing;
        maxDepth = trainContext.strategy.maxDepth;
    }

    int[][] generateNodeToFeatures(
            List<LearningNode> layer, Consumer<int[]> nodeFeaturePairsSetter) {
        int numNodes = layer.size();
        // Generates (nodeId, featureId) pairs that are required to build histograms.
        int[][] nodeToFeatures = new int[numNodes][];
        IntArrayList nodeFeaturePairs = new IntArrayList(numNodes * numBaggingFeatures * 2);
        for (int k = 0; k < numNodes; k += 1) {
            LearningNode node = layer.get(k);
            if (node.depth == maxDepth) {
                // Ignores the results, just to consume the randomizer.
                DataUtils.sample(featureIndicesPool, numBaggingFeatures, featureRandomizer);
                // No need to calculate histograms for features, only sum of gradients and hessians
                // are needed. Uses `numFeatures` to indicate this special "feature".
                nodeToFeatures[k] = new int[] {numFeatures};
            } else {
                nodeToFeatures[k] =
                        DataUtils.sample(featureIndicesPool, numBaggingFeatures, featureRandomizer);
                Arrays.sort(nodeToFeatures[k]);
            }
            for (int featureId : nodeToFeatures[k]) {
                nodeFeaturePairs.add(k);
                nodeFeaturePairs.add(featureId);
            }
        }
        nodeFeaturePairsSetter.accept(nodeFeaturePairs.toArray());
        return nodeToFeatures;
    }

    /** Calculate local histograms for nodes in current layer of tree. */
    void build(
            List<LearningNode> layer,
            int[] indices,
            BinnedInstance[] instances,
            double[] pgh,
            Consumer<int[]> nodeFeaturePairsSetter,
            ExecutorService executor,
            int numThreads,
            CacheDataCalcLocalHistsOperator.OutputQueuedCollector collector) {
        LOG.info("subtaskId: {}, {} start", subtaskId, HistBuilder.class.getSimpleName());

        // Generates (nodeId, featureId) pairs that are required to build histograms.
        int[][] nodeToFeatures = generateNodeToFeatures(layer, nodeFeaturePairsSetter);

        // Calculates histograms for (nodeId, featureId) pairs.
        HistBuilderImpl builderImpl =
                new HistBuilderImpl(
                        layer,
                        maxDepth,
                        numFeatures,
                        numFeatureBins,
                        nodeToFeatures,
                        indices,
                        instances,
                        pgh);
        builderImpl.init(isUnseenMissing, featureMetas);
        builderImpl.calcHistsForPairs(subtaskId, executor, numThreads, collector);

        LOG.info("subtaskId: {}, {} end", subtaskId, HistBuilder.class.getSimpleName());
    }

    /** Calculate local histograms for nodes in current layer of tree. */
    void build(
            List<LearningNode> layer,
            int[] indices,
            BinnedInstance[] instances,
            double[] pgh,
            Consumer<int[]> nodeFeaturePairsSetter,
            Collector<Tuple5<Integer, Integer, Integer, Integer, Histogram>> collector) {
        LOG.info("subtaskId: {}, {} start", subtaskId, HistBuilder.class.getSimpleName());

        // Generates (nodeId, featureId) pairs that are required to build histograms.
        int[][] nodeToFeatures = generateNodeToFeatures(layer, nodeFeaturePairsSetter);

        // Calculates histograms for (nodeId, featureId) pairs.
        HistBuilderImpl builderImpl =
                new HistBuilderImpl(
                        layer,
                        maxDepth,
                        numFeatures,
                        numFeatureBins,
                        nodeToFeatures,
                        indices,
                        instances,
                        pgh);
        builderImpl.init(isUnseenMissing, featureMetas);
        builderImpl.calcHistsForPairs(subtaskId, collector);

        LOG.info("subtaskId: {}, {} end", subtaskId, HistBuilder.class.getSimpleName());
    }

    static class HistBuilderImpl {
        private final List<LearningNode> layer;
        private final int maxDepth;
        private final int numFeatures;
        private final int[] numFeatureBins;
        private final int[][] nodeToFeatures;
        private final int[] indices;
        private final BinnedInstance[] instances;
        private final double[] pgh;

        private int[] featureDefaultVal;

        public HistBuilderImpl(
                List<LearningNode> layer,
                int maxDepth,
                int numFeatures,
                int[] numFeatureBins,
                int[][] nodeToFeatures,
                int[] indices,
                BinnedInstance[] instances,
                double[] pgh) {
            this.layer = layer;
            this.maxDepth = maxDepth;
            this.numFeatures = numFeatures;
            this.numFeatureBins = numFeatureBins;
            this.nodeToFeatures = nodeToFeatures;
            this.indices = indices;
            this.instances = instances;
            this.pgh = pgh;
            Preconditions.checkArgument(numFeatureBins.length == numFeatures + 1);
        }

        private static void calcHistsForDefaultBin(
                int defaultVal,
                int featureOffset,
                int numBins,
                double[] totalHists,
                double[] hists) {
            int defaultValIndex = (featureOffset + defaultVal) * BIN_SIZE;
            hists[defaultValIndex] = totalHists[0];
            hists[defaultValIndex + 1] = totalHists[1];
            hists[defaultValIndex + 2] = totalHists[2];
            hists[defaultValIndex + 3] = totalHists[3];
            for (int i = 0; i < numBins; i += 1) {
                if (i != defaultVal) {
                    int index = (featureOffset + i) * BIN_SIZE;
                    add(
                            hists,
                            featureOffset,
                            defaultVal,
                            -hists[index],
                            -hists[index + 1],
                            -hists[index + 2],
                            -hists[index + 3]);
                }
            }
        }

        private static void add(
                double[] hists, int offset, int val, double d0, double d1, double d2, double d3) {
            int index = (offset + val) * BIN_SIZE;
            hists[index] += d0;
            hists[index + 1] += d1;
            hists[index + 2] += d2;
            hists[index + 3] += d3;
        }

        private void init(boolean isUnseenMissing, FeatureMeta[] featureMetas) {
            featureDefaultVal = new int[numFeatures];
            for (int i = 0; i < numFeatures; i += 1) {
                FeatureMeta d = featureMetas[i];
                featureDefaultVal[i] =
                        !isUnseenMissing && d instanceof FeatureMeta.ContinuousFeatureMeta
                                ? ((FeatureMeta.ContinuousFeatureMeta) d).zeroBin
                                : d.missingBin;
            }
        }

        private void calcTotalHists(int start, int end, double[] totalHists) {
            for (int i = start; i < end; i += 1) {
                int instanceId = indices[i];
                BinnedInstance binnedInstance = instances[instanceId];
                double weight = binnedInstance.weight;
                double gradient = pgh[3 * instanceId + 1];
                double hessian = pgh[3 * instanceId + 2];
                add(totalHists, 0, 0, gradient, hessian, weight, 1.);
            }
        }

        private void calcHistsForNonDefaultBins(
                int start,
                int end,
                boolean allFeatureValid,
                BitSet featureValid,
                int[] featureOffset,
                double[] hists) {
            for (int i = start; i < end; i += 1) {
                int instanceId = indices[i];
                BinnedInstance binnedInstance = instances[instanceId];
                double weight = binnedInstance.weight;
                double gradient = pgh[3 * instanceId + 1];
                double hessian = pgh[3 * instanceId + 2];

                if (null == binnedInstance.featureIds) {
                    for (int j = 0; j < binnedInstance.featureValues.length; j += 1) {
                        if (allFeatureValid || featureValid.get(j)) {
                            add(
                                    hists,
                                    featureOffset[j],
                                    binnedInstance.featureValues[j],
                                    gradient,
                                    hessian,
                                    weight,
                                    1.);
                        }
                    }
                } else {
                    for (int j = 0; j < binnedInstance.featureIds.length; j += 1) {
                        int featureId = binnedInstance.featureIds[j];
                        if (allFeatureValid || featureValid.get(featureId)) {
                            add(
                                    hists,
                                    featureOffset[featureId],
                                    binnedInstance.featureValues[j],
                                    gradient,
                                    hessian,
                                    weight,
                                    1.);
                        }
                    }
                }
            }
        }

        private void calcHistsForSplitNode(
                int start, int end, int[] features, int[] binOffsets, double[] hists) {
            double[] totalHists = new double[4];
            calcTotalHists(start, end, totalHists);

            int[] featureOffsets = new int[numFeatures];
            BitSet featureValid = null;
            boolean allFeatureValid;
            if (numFeatures != features.length) {
                allFeatureValid = false;
                featureValid = new BitSet(numFeatures);
                for (int i = 0; i < features.length; i += 1) {
                    featureValid.set(features[i]);
                    featureOffsets[features[i]] = binOffsets[i];
                }
            } else {
                allFeatureValid = true;
                System.arraycopy(binOffsets, 0, featureOffsets, 0, numFeatures);
            }

            calcHistsForNonDefaultBins(
                    start, end, allFeatureValid, featureValid, featureOffsets, hists);

            for (int featureId : features) {
                calcHistsForDefaultBin(
                        featureDefaultVal[featureId],
                        featureOffsets[featureId],
                        numFeatureBins[featureId],
                        totalHists,
                        hists);
            }
        }

        /** Calculate histograms for all (nodeId, featureId) pairs. */
        private void calcHistsForPairs(
                int subtaskId,
                ExecutorService executor,
                int numThreads,
                CacheDataCalcLocalHistsOperator.OutputQueuedCollector collector) {
            long start = System.currentTimeMillis();
            int numNodes = layer.size();

            final int sliceSize = (1 << 15) / numThreads;
            List<Future<Void>> futures = new ArrayList<>();
            collector.start();
            int offset = 0;
            int pairBaseId = 0;
            for (int k = 0; k < numNodes; k += 1) {
                int nodeId = k;
                int[] features = nodeToFeatures[nodeId];
                final int nodeOffset = offset;
                int[] binOffsets = new int[features.length];
                for (int i = 0; i < features.length; i += 1) {
                    binOffsets[i] = offset - nodeOffset;
                    offset += numFeatureBins[features[i]];
                }
                LearningNode node = layer.get(k);

                int numSlices = (node.slice.size() - 1) / sliceSize + 1;
                int finalOffset = offset;
                int finalPairBaseId = pairBaseId;

                for (int i = 0; i < numSlices; i += 1) {
                    int sliceId = i;
                    int sliceStart = node.slice.start + i * sliceSize;
                    int sliceEnd = Math.min(sliceStart + sliceSize, node.slice.end);
                    Future<Void> future =
                            executor.submit(
                                    () -> {
                                        double[] hists =
                                                new double[(finalOffset - nodeOffset) * BIN_SIZE];
                                        long nodeSubtaskStart = System.currentTimeMillis();
                                        if (node.depth != maxDepth) {
                                            calcHistsForSplitNode(
                                                    sliceStart,
                                                    sliceEnd,
                                                    features,
                                                    binOffsets,
                                                    hists);
                                        } else {
                                            calcTotalHists(sliceStart, sliceEnd, hists);
                                        }
                                        LOG.info(
                                                "subtaskId: {}, node {}, {} #instances, {} #features, {} ms",
                                                subtaskId,
                                                nodeId,
                                                sliceEnd - sliceStart,
                                                features.length,
                                                System.currentTimeMillis() - nodeSubtaskStart);

                                        int featureBinStart = 0;
                                        for (int j = 0; j < features.length; j += 1) {
                                            int featureNumBins =
                                                    numFeatureBins[features[j]] * BIN_SIZE;
                                            Slice featureBinSlice =
                                                    new Slice(
                                                            featureBinStart,
                                                            featureBinStart + featureNumBins);
                                            int pairId = finalPairBaseId + j;
                                            collector.collect(
                                                    Tuple5.of(
                                                            subtaskId,
                                                            pairId,
                                                            sliceId,
                                                            numSlices,
                                                            new Histogram(hists, featureBinSlice)));
                                            featureBinStart += featureNumBins;
                                        }
                                        return null;
                                    });
                    futures.add(future);
                }
                pairBaseId += features.length;
            }
            try {
                for (Future<Void> future : futures) {
                    future.get();
                }
                collector.awaitTermination();
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
            LOG.info(
                    "subtaskId: {}, elapsed time for calculating histograms: {} ms",
                    subtaskId,
                    System.currentTimeMillis() - start);
        }

        private void calcHistsForPairs(
                int subtaskId,
                Collector<Tuple5<Integer, Integer, Integer, Integer, Histogram>> out) {
            long start = System.currentTimeMillis();
            int numNodes = layer.size();
            int offset = 0;
            int pairBaseId = 0;
            for (int k = 0; k < numNodes; k += 1) {
                int[] features = nodeToFeatures[k];
                final int nodeOffset = offset;
                int[] binOffsets = new int[features.length];
                for (int i = 0; i < features.length; i += 1) {
                    binOffsets[i] = offset - nodeOffset;
                    offset += numFeatureBins[features[i]];
                }

                double[] nodeHists = new double[(offset - nodeOffset) * BIN_SIZE];
                long nodeStart = System.currentTimeMillis();
                LearningNode node = layer.get(k);
                if (node.depth != maxDepth) {
                    calcHistsForSplitNode(
                            node.slice.start, node.slice.end, features, binOffsets, nodeHists);
                } else {
                    calcTotalHists(node.slice.start, node.slice.end, nodeHists);
                }
                LOG.info(
                        "subtaskId: {}, node {}, {} #instances, {} #features, {} ms",
                        subtaskId,
                        k,
                        node.slice.size(),
                        features.length,
                        System.currentTimeMillis() - nodeStart);

                int sliceStart = 0;
                for (int i = 0; i < features.length; i += 1) {
                    int sliceSize = numFeatureBins[features[i]] * BIN_SIZE;
                    int pairId = pairBaseId + i;
                    out.collect(
                            Tuple5.of(
                                    subtaskId,
                                    pairId,
                                    0,
                                    1,
                                    new Histogram(
                                            nodeHists,
                                            new Slice(sliceStart, sliceStart + sliceSize))));
                    sliceStart += sliceSize;
                }
                pairBaseId += features.length;
            }

            LOG.info(
                    "subtaskId: {}, elapsed time for calculating histograms: {} ms",
                    subtaskId,
                    System.currentTimeMillis() - start);
        }
    }
}
