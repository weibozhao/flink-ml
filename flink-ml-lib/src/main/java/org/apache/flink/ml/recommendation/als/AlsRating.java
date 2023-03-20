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

import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

import java.io.IOException;
import java.util.HashMap;
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
public class AlsRating implements Estimator<AlsRating, AlsModel>, AlsParams<AlsRating> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public AlsRating() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public AlsModel fit(Table... inputs) {
        Table modelData = new AlsKernel(paramMap).fit(inputs[0]);
        AlsModel model = new AlsModel().setModelData(modelData);
        //ReadWriteUtils.updateExistingParams(model, paramMap);
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static AlsRating load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
