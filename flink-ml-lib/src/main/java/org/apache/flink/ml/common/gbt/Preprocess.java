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

import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.gbt.defs.BoostingStrategy;
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizer;
import org.apache.flink.ml.feature.stringindexer.StringIndexer;
import org.apache.flink.ml.feature.stringindexer.StringIndexerModel;
import org.apache.flink.ml.feature.stringindexer.StringIndexerModelData;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.ApiExpression;
import org.apache.flink.table.api.Expressions;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;
import org.eclipse.collections.impl.list.mutable.primitive.DoubleArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import static org.apache.flink.table.api.Expressions.$;

/**
 * Preprocesses input data table for gradient boosting trees algorithms.
 *
 * <p>Multiple non-vector columns or a single vector column can be specified for preprocessing.
 * Values of these column(s) are mapped to integers inplace through discretizer or string indexer,
 * and the meta information of column(s) are obtained.
 */
class Preprocess {
    private static final Logger LOG = LoggerFactory.getLogger(Preprocess.class);

    /**
     * Maps continuous and categorical columns to integers inplace using quantile discretizer and
     * string indexer respectively, and obtains meta information for all columns.
     */
    static Tuple2<Table, DataStream<FeatureMeta>> preprocessCols(
            Table dataTable, BoostingStrategy strategy) {

        String[] relatedCols = ArrayUtils.add(strategy.featuresCols, strategy.labelCol);
        if (null != strategy.weightCol) {
            relatedCols = ArrayUtils.add(relatedCols, strategy.weightCol);
        }
        dataTable =
                dataTable.select(
                        Arrays.stream(relatedCols)
                                .map(Expressions::$)
                                .toArray(ApiExpression[]::new));

        // Maps continuous columns to integers, and obtain corresponding discretizer model.
        String[] continuousCols =
                ArrayUtils.removeElements(strategy.featuresCols, strategy.categoricalCols);
        Tuple2<Table, DataStream<double[][]>> continuousMappedDataAndModelData =
                discretizeContinuousCols(
                        dataTable, continuousCols, strategy.maxBins, strategy.seed);
        dataTable = continuousMappedDataAndModelData.f0;
        DataStream<FeatureMeta> continuousFeatureMeta =
                buildContinuousFeatureMeta(
                        continuousMappedDataAndModelData.f1,
                        continuousCols,
                        strategy.isInputVector);

        // Maps categorical columns to integers, and obtain string indexer model.
        DataStream<FeatureMeta> categoricalFeatureMeta;
        if (strategy.categoricalCols.length > 0) {
            String[] mappedCategoricalCols =
                    Arrays.stream(strategy.categoricalCols)
                            .map(d -> d + "_output")
                            .toArray(String[]::new);
            StringIndexer stringIndexer =
                    new StringIndexer()
                            .setInputCols(strategy.categoricalCols)
                            .setOutputCols(mappedCategoricalCols)
                            .setHandleInvalid("keep")
                            .setMaxIndexNum(strategy.maxCategoriesNum)
                            .setStringOrderType("frequencyDesc");
            StringIndexerModel stringIndexerModel = stringIndexer.fit(dataTable);
            dataTable = stringIndexerModel.transform(dataTable)[0];

            categoricalFeatureMeta =
                    buildCategoricalFeatureMeta(
                            StringIndexerModelData.getModelDataStream(
                                    stringIndexerModel.getModelData()[0]),
                            strategy.categoricalCols);
        } else {
            categoricalFeatureMeta =
                    continuousFeatureMeta
                            .<FeatureMeta>flatMap((value, out) -> {})
                            .returns(TypeInformation.of(FeatureMeta.class));
        }

        // Rename results columns.
        ApiExpression[] dropColumnExprs =
                Arrays.stream(strategy.categoricalCols)
                        .map(Expressions::$)
                        .toArray(ApiExpression[]::new);
        ApiExpression[] renameColumnExprs =
                Arrays.stream(strategy.categoricalCols)
                        .map(d -> $(d + "_output").as(d))
                        .toArray(ApiExpression[]::new);
        dataTable = dataTable.dropColumns(dropColumnExprs).renameColumns(renameColumnExprs);

        return Tuple2.of(dataTable, continuousFeatureMeta.union(categoricalFeatureMeta));
    }

    /**
     * Maps features values in vectors to indices using quantile discretizer, and obtains meta
     * information for all features.
     */
    static Tuple2<Table, DataStream<FeatureMeta>> preprocessVecCol(
            Table dataTable, BoostingStrategy strategy) {
        dataTable =
                null == strategy.weightCol
                        ? dataTable.select($(strategy.featuresCols[0]), $(strategy.labelCol))
                        : dataTable.select(
                                $(strategy.featuresCols[0]),
                                $(strategy.labelCol),
                                $(strategy.weightCol));
        Tuple2<Table, DataStream<double[][]>> mappedDataAndBinEdges =
                discretizeVectorCol(
                        dataTable, strategy.featuresCols[0], strategy.maxBins, strategy.seed);
        return Tuple2.of(
                mappedDataAndBinEdges.f0,
                buildContinuousFeatureMeta(
                        mappedDataAndBinEdges.f1, strategy.featuresCols, strategy.isInputVector));
    }

    /** Builds {@link FeatureMeta} from {@link StringIndexerModelData}. */
    private static DataStream<FeatureMeta> buildCategoricalFeatureMeta(
            DataStream<StringIndexerModelData> stringIndexerModelData, String[] cols) {
        return stringIndexerModelData
                .<FeatureMeta>flatMap(
                        (d, out) -> {
                            Preconditions.checkArgument(d.stringArrays.length == cols.length);
                            LOG.info(
                                    "#categories for {}: {}",
                                    cols,
                                    Arrays.stream(d.stringArrays)
                                            .map(arr -> arr.length)
                                            .collect(Collectors.toList()));
                            for (int i = 0; i < cols.length; i += 1) {
                                out.collect(
                                        FeatureMeta.categorical(
                                                cols[i],
                                                d.stringArrays[i].length,
                                                d.stringArrays[i]));
                            }
                        })
                .returns(TypeInformation.of(FeatureMeta.class));
    }

    /** Builds {@link FeatureMeta} from bin edges. */
    private static DataStream<FeatureMeta> buildContinuousFeatureMeta(
            DataStream<double[][]> discretizerModelData, String[] cols, boolean isInputVector) {
        // Column name template for vector case
        final String vectorColNameTemplate = "_%s_f%d";
        return discretizerModelData
                .<FeatureMeta>flatMap(
                        (d, out) -> {
                            for (int i = 0; i < d.length; i += 1) {
                                String name =
                                        (!isInputVector)
                                                ? cols[i]
                                                : String.format(vectorColNameTemplate, cols[0], i);
                                out.collect(FeatureMeta.continuous(name, d[i].length - 1, d[i]));
                            }
                        })
                .returns(TypeInformation.of(FeatureMeta.class));
    }

    /** Discretizes continuous columns inplace, and obtains quantile discretizer model data. */
    private static Tuple2<Table, DataStream<double[][]>> discretizeContinuousCols(
            Table dataTable, String[] continuousCols, int numBins, long seed) {
        final StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) dataTable).getTableEnvironment();
        final int nCols = continuousCols.length;

        // Merges all continuous columns into a vector columns.
        final String vectorCol = "_vec";
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(dataTable.getResolvedSchema());
        DataStream<Row> data = tEnv.toDataStream(dataTable, Row.class);
        DataStream<Row> dataWithVectors =
                data.map(
                        (row) -> {
                            double[] values = new double[nCols];
                            for (int i = 0; i < nCols; i += 1) {
                                Number number = row.getFieldAs(continuousCols[i]);
                                // Null values are represented using `Double.NaN` in `DenseVector`.
                                values[i] = (null == number) ? Double.NaN : number.doubleValue();
                            }
                            return Row.join(row, Row.of(Vectors.dense(values)));
                        },
                        new RowTypeInfo(
                                ArrayUtils.add(
                                        inputTypeInfo.getFieldTypes(),
                                        DenseVectorTypeInfo.INSTANCE),
                                ArrayUtils.add(inputTypeInfo.getFieldNames(), vectorCol)));

        Tuple2<Table, DataStream<double[][]>> mappedDataAndModelData =
                discretizeVectorCol(tEnv.fromDataStream(dataWithVectors), vectorCol, numBins, seed);
        DataStream<Row> discretized = tEnv.toDataStream(mappedDataAndModelData.f0, Row.class);

        // Maps the result vector back to multiple continuous columns.
        final String[] otherCols =
                ArrayUtils.removeElements(inputTypeInfo.getFieldNames(), continuousCols);
        final TypeInformation<?>[] otherColTypes =
                Arrays.stream(otherCols)
                        .map(inputTypeInfo::getTypeAt)
                        .toArray(TypeInformation[]::new);
        final TypeInformation<?>[] mappedColTypes =
                Arrays.stream(continuousCols).map(d -> Types.INT).toArray(TypeInformation[]::new);

        DataStream<Row> mapped =
                discretized.map(
                        (row) -> {
                            DenseVector vec = row.getFieldAs(vectorCol);
                            Integer[] ints =
                                    Arrays.stream(vec.values)
                                            .mapToObj(d -> (Integer) ((int) d))
                                            .toArray(Integer[]::new);
                            Row result = Row.project(row, otherCols);
                            for (int i = 0; i < ints.length; i += 1) {
                                result.setField(continuousCols[i], ints[i]);
                            }
                            return result;
                        },
                        new RowTypeInfo(
                                ArrayUtils.addAll(otherColTypes, mappedColTypes),
                                ArrayUtils.addAll(otherCols, continuousCols)));

        return Tuple2.of(tEnv.fromDataStream(mapped), mappedDataAndModelData.f1);
    }

    /**
     * Discretize the vector column inplace using quantile discretizer, and obtains quantile
     * discretizer model data. The computation is similar to {@link KBinsDiscretizer} with
     * `QUANTILE` strategy, except that unseen entries in sparse vectors are kept unchanged.
     */
    private static Tuple2<Table, DataStream<double[][]>> discretizeVectorCol(
            Table dataTable, String vectorCol, int numBins, long seed) {
        final StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) dataTable).getTableEnvironment();
        DataStream<Row> data = tEnv.toDataStream(dataTable, Row.class);

        final int numSamples = 50000;
        DataStream<Tuple3<Integer, Double, Integer>> entries =
                DataStreamUtils.sample(
                                data.map(d -> d.getFieldAs(vectorCol), VectorTypeInfo.INSTANCE),
                                numSamples,
                                seed)
                        .flatMap(
                                (d, out) -> {
                                    if (d instanceof DenseVector) {
                                        DenseVector dv = (DenseVector) d;
                                        for (int i = 0; i < dv.size(); i += 1) {
                                            out.collect(Tuple3.of(i, dv.get(i), d.size()));
                                        }
                                    } else {
                                        SparseVector sv = (SparseVector) d;
                                        for (int i = 0; i < sv.indices.length; i += 1) {
                                            out.collect(
                                                    Tuple3.of(
                                                            sv.indices[i], sv.values[i], d.size()));
                                        }
                                    }
                                },
                                Types.TUPLE(Types.INT, Types.DOUBLE, Types.INT));

        DataStream<Tuple3<Integer, double[], Integer>> columnBinEdges =
                DataStreamUtils.mapPartition(
                        entries.keyBy(value -> value.f0), new CalcBinEdgesFunction(numBins));
        DataStream<double[][]> binEdges =
                DataStreamUtils.mapPartition(
                        columnBinEdges,
                        (values, out) -> {
                            double[][] binEdgesArr = null;
                            for (Tuple3<Integer, double[], Integer> value : values) {
                                if (null == binEdgesArr) {
                                    binEdgesArr = new double[value.f2][];
                                }
                                binEdgesArr[value.f0] = value.f1;
                            }
                            if (null == binEdgesArr) {
                                binEdgesArr = new double[0][];
                            }
                            for (int i = 0; i < binEdgesArr.length; i += 1) {
                                if (null == binEdgesArr[i]) {
                                    binEdgesArr[i] =
                                            new double[] {
                                                Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY
                                            };
                                }
                            }
                            out.collect(binEdgesArr);
                        },
                        Types.OBJECT_ARRAY(
                                PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO));
        binEdges.getTransformation().setParallelism(1);

        final String broadcastModelKey = "broadcastModelKey";
        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(data),
                        Collections.singletonMap(broadcastModelKey, binEdges),
                        inputList -> {
                            //noinspection unchecked
                            DataStream<Row> input = (DataStream<Row>) inputList.get(0);
                            return input.map(
                                    new FindBinFunction(vectorCol, broadcastModelKey),
                                    TableUtils.getRowTypeInfo(dataTable.getResolvedSchema()));
                        });
        return Tuple2.of(tEnv.fromDataStream(output), binEdges);
    }

    /**
     * Calculate bin edges from entries. The input elements are (column index, column value), and
     * the output elements are (column index, bin edges).
     */
    private static class CalcBinEdgesFunction
            implements MapPartitionFunction<
                    Tuple3<Integer, Double, Integer>, Tuple3<Integer, double[], Integer>> {
        private final int numBins;

        public CalcBinEdgesFunction(int numBins) {
            this.numBins = numBins;
        }

        @Override
        public void mapPartition(
                Iterable<Tuple3<Integer, Double, Integer>> values,
                Collector<Tuple3<Integer, double[], Integer>> out) {
            Map<Integer, DoubleArrayList> columnElementsMap = new HashMap<>();
            int vectorSize = 0;
            for (Tuple3<Integer, Double, Integer> value : values) {
                if (vectorSize <= 0) {
                    vectorSize = value.f2;
                }
                DoubleArrayList elements =
                        columnElementsMap.compute(
                                value.f0, (k, v) -> null == v ? new DoubleArrayList() : v);
                if (!Double.isNaN(value.f1)) {
                    elements.add(value.f1);
                }
            }
            for (int columnId : columnElementsMap.keySet()) {
                double[] elements = columnElementsMap.get(columnId).toArray();
                Arrays.sort(elements);
                if (elements[0] == elements[elements.length - 1]) {
                    LOG.warn("Feature {} is constant and the output will all be zero.", columnId);
                    out.collect(
                            Tuple3.of(
                                    columnId,
                                    new double[] {
                                        Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY
                                    },
                                    vectorSize));
                } else {
                    out.collect(
                            Tuple3.of(
                                    columnId,
                                    KBinsDiscretizer.removeDuplicatedBinEdges(elements, numBins),
                                    vectorSize));
                }
            }
        }
    }

    private static class FindBinFunction extends RichMapFunction<Row, Row> {
        private final String vectorCol;
        private final String broadcastKey;
        /** Model data used to find bins for each feature. */
        private double[][] binEdges;

        private FindBinFunction(String vectorCol, String broadcastKey) {
            this.vectorCol = vectorCol;
            this.broadcastKey = broadcastKey;
        }

        @Override
        public Row map(Row value) {
            if (null == binEdges) {
                binEdges =
                        getRuntimeContext().<double[][]>getBroadcastVariable(broadcastKey).get(0);
            }
            Vector vec = value.getFieldAs(vectorCol);
            if (vec instanceof DenseVector) {
                DenseVector dv = (DenseVector) vec;
                for (int i = 0; i < dv.size(); i += 1) {
                    double v = dv.get(i);
                    if (Double.isNaN(v)) {
                        dv.set(i, binEdges[i].length - 1);
                    } else {
                        dv.set(i, DataUtils.findBin(binEdges[i], v));
                    }
                }
            } else {
                SparseVector sv = (SparseVector) vec;
                for (int i = 0; i < sv.indices.length; i += 1) {
                    double v = sv.values[i];
                    if (Double.isNaN(v)) {
                        sv.set(i, binEdges[i].length - 1);
                    } else {
                        sv.values[i] = DataUtils.findBin(binEdges[sv.indices[i]], v);
                    }
                }
            }
            return value;
        }
    }
}
