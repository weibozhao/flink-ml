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

package org.apache.flink.ml.common.gbt;

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.api.common.typeinfo.TypeInfo;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.classification.gbtclassifier.GBTClassifierModelServable;
import org.apache.flink.ml.common.gbt.defs.Node;
import org.apache.flink.ml.common.gbt.typeinfo.GBTModelDataSerializer;
import org.apache.flink.ml.common.gbt.typeinfo.GBTModelDataTypeInfoFactory;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;

import org.eclipse.collections.impl.map.mutable.primitive.IntDoubleHashMap;
import org.eclipse.collections.impl.map.mutable.primitive.IntObjectHashMap;
import org.eclipse.collections.impl.map.mutable.primitive.ObjectIntHashMap;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.BitSet;
import java.util.List;
import java.util.function.Function;

/**
 * Model data of {@link GBTClassifierModelServable}.
 *
 * <p>This class also provides methods to convert model data from Table to Datastream, and classes
 * to save/load model data.
 */
@TypeInfo(GBTModelDataTypeInfoFactory.class)
public class GBTModelData {

    public String type;
    public boolean isInputVector;

    public double prior;
    public double stepSize;

    public List<List<Node>> allTrees;
    public List<String> featureNames;
    public IntObjectHashMap<ObjectIntHashMap<String>> categoryToIdMaps;
    public IntObjectHashMap<double[]> featureIdToBinEdges;
    public BitSet isCategorical;

    public GBTModelData() {}

    public GBTModelData(
            String type,
            boolean isInputVector,
            double prior,
            double stepSize,
            List<List<Node>> allTrees,
            List<String> featureNames,
            IntObjectHashMap<ObjectIntHashMap<String>> categoryToIdMaps,
            IntObjectHashMap<double[]> featureIdToBinEdges,
            BitSet isCategorical) {
        this.type = type;
        this.isInputVector = isInputVector;
        this.prior = prior;
        this.stepSize = stepSize;
        this.allTrees = allTrees;
        this.featureNames = featureNames;
        this.categoryToIdMaps = categoryToIdMaps;
        this.featureIdToBinEdges = featureIdToBinEdges;
        this.isCategorical = isCategorical;
    }

    /** The mapping computation is from StringIndexerModel. */
    private static int mapCategoricalFeature(ObjectIntHashMap<String> categoryToId, Object v) {
        String s;
        if (v instanceof String) {
            s = (String) v;
        } else if (v instanceof Number) {
            s = String.valueOf(v);
        } else if (null == v) {
            s = null;
        } else {
            throw new RuntimeException("Categorical column only supports string and numeric type.");
        }
        return categoryToId.getIfAbsent(s, categoryToId.size());
    }

    /**
     * Reads and deserializes the model data from the input stream.
     *
     * @param inputStream The stream to read from.
     * @return The model data instance.
     */
    public static GBTModelData decode(InputStream inputStream) throws IOException {
        DataInputViewStreamWrapper dataInputViewStreamWrapper =
                new DataInputViewStreamWrapper(inputStream);
        GBTModelDataSerializer serializer = new GBTModelDataSerializer();
        return serializer.deserialize(dataInputViewStreamWrapper);
    }

    /**
     * Constructs features from arbitrary objects.
     *
     * @param featureKeys Keys of features.
     * @param getFeature A function to get feature values.
     * @return The features.
     * @param <K> The type of feature keys.
     */
    public <K> IntDoubleHashMap toFeatures(K[] featureKeys, Function<K, Object> getFeature) {
        IntDoubleHashMap features = new IntDoubleHashMap();
        if (isInputVector) {
            // TODO: fix this after Designer supports storing meta information.
            Object obj = getFeature.apply(featureKeys[0]);
            SparseVector sv =
                    obj instanceof String
                            ? parseLibSVMStr((String) obj, Integer.MAX_VALUE)
                            : ((Vector) obj).toSparse();
            for (int i = 0; i < sv.indices.length; i += 1) {
                features.put(sv.indices[i], sv.values[i]);
            }
        } else {
            for (int i = 0; i < featureKeys.length; i += 1) {
                Object obj = getFeature.apply(featureKeys[i]);
                double v;
                if (isCategorical.get(i)) {
                    v = mapCategoricalFeature(categoryToIdMaps.get(i), obj);
                } else {
                    Number number = (Number) obj;
                    v = (null == number) ? Double.NaN : number.doubleValue();
                }
                features.put(i, v);
            }
        }
        return features;
    }

    static SparseVector parseLibSVMStr(String s, int size) {
        String[] split = s.split("[ :]");
        int numElems = split.length / 2;
        int[] indices = new int[numElems];
        double[] values = new double[numElems];
        for (int i = 0; i < numElems; i += 1) {
            indices[i] = Integer.parseInt(split[2 * i]);
            values[i] = Double.parseDouble(split[2 * i + 1]);
        }
        return Vectors.sparse(size, indices, values);
    }

    /**
     * Get raw prediction for the features.
     *
     * @param rawFeatures The features.
     * @return Raw prediction.
     */
    public double predictRaw(IntDoubleHashMap rawFeatures) {
        double v = prior;
        for (List<Node> treeNodes : allTrees) {
            Node node = treeNodes.get(0);
            while (!node.isLeaf) {
                boolean goLeft = node.split.shouldGoLeft(rawFeatures);
                node = goLeft ? treeNodes.get(node.left) : treeNodes.get(node.right);
            }
            v += stepSize * node.split.prediction;
        }
        return v;
    }

    /**
     * Serializes the instance and writes to the output stream.
     *
     * @param outputStream The stream to write to.
     */
    @VisibleForTesting
    public void encode(OutputStream outputStream) throws IOException {
        DataOutputViewStreamWrapper dataOutputViewStreamWrapper =
                new DataOutputViewStreamWrapper(outputStream);
        GBTModelDataSerializer serializer = new GBTModelDataSerializer();
        serializer.serialize(this, dataOutputViewStreamWrapper);
    }

    @Override
    public String toString() {
        return String.format(
                "GBTModelData{type=%s, prior=%s, allTrees=%s, categoryToIdMaps=%s, featureIdToBinEdges=%s, isCategorical=%s}",
                type, prior, allTrees, categoryToIdMaps, featureIdToBinEdges, isCategorical);
    }
}
