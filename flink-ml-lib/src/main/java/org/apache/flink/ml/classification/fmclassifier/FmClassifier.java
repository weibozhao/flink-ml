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

package org.apache.flink.ml.classification.fmclassifier;

import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.fm.BaseFmTrain;
import org.apache.flink.ml.common.ps.api.MLData;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/** An Estimator which implements the fm classifier algorithm. */
public class FmClassifier
        implements Estimator<FmClassifier, FmClassifierModel>, FmClassifierParams<FmClassifier> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public FmClassifier() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public FmClassifierModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        MLData modelData =
                new BaseFmTrain(false, paramMap).train(MLData.of(inputs, new String[] {"data"}));
        FmClassifierModel model = new FmClassifierModel().setModelData(modelData.getTable());
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static FmClassifier load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
