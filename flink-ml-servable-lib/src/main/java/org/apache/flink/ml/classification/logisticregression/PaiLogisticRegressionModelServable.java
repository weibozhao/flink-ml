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

package org.apache.flink.ml.classification.logisticregression;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.servable.api.PaiModelServable;
import org.apache.flink.ml.servable.api.Row;
import org.apache.flink.ml.servable.types.BasicType;
import org.apache.flink.ml.servable.types.DataType;
import org.apache.flink.ml.servable.types.DataTypes;

import java.io.IOException;
import java.util.List;

/** A Servable which can be used to classifies data in online inference. */
public class PaiLogisticRegressionModelServable extends LogisticRegressionModelServable
        implements PaiModelServable<PaiLogisticRegressionModelServable> {

    public PaiLogisticRegressionModelServable() {
        super();
    }

    PaiLogisticRegressionModelServable(LogisticRegressionModelData modelData) {
        super(modelData);
    }

    @Override
    public Tuple2<String[], DataType[]> getResultNamesAndTypes() {
        return Tuple2.of(
                new String[] {getPredictionCol(), getRawPredictionCol()},
                new DataType[] {DataTypes.DOUBLE, DataTypes.VECTOR(BasicType.DOUBLE)});
    }

    @Override
    public PaiLogisticRegressionModelServable setModelData(List<Row> modelRows) throws IOException {
        DenseVector vec = (DenseVector) modelRows.get(0).get(0);
        setModelData(new LogisticRegressionModelData(vec, 1L));
        return this;
    }
}
