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

import org.apache.flink.ml.common.lossfunc.LossFunc;
import org.apache.flink.ml.common.optimizer.Optimizer;
import org.apache.flink.ml.common.optimizer.SGD;
import org.apache.flink.ml.common.param.HasElasticNet;
import org.apache.flink.ml.common.param.HasGlobalBatchSize;
import org.apache.flink.ml.common.param.HasLearningRate;
import org.apache.flink.ml.common.param.HasMaxIter;
import org.apache.flink.ml.common.param.HasReg;
import org.apache.flink.ml.common.param.HasTol;
import org.apache.flink.ml.common.ps.iterations.HighComponent;
import org.apache.flink.ml.param.Param;
import org.apache.flink.streaming.api.datastream.DataStream;

import java.util.Map;

/** Map Stage. */
public class OptimizerComponent extends HighComponent {

    private final int maxIter;
    private final double learningRate;
    private final int globalBatchSize;
    private final double tol;
    private final double reg;
    private final double elasticNet;
    private final LossFunc lossFunc;
    private String initModel;
    private final Method method;

    public OptimizerComponent(Map<Param<?>, Object> paramMap, Method method, LossFunc lossFunc) {
        this.maxIter = (int) paramMap.get(HasMaxIter.MAX_ITER);
        this.learningRate = (double) paramMap.get(HasLearningRate.LEARNING_RATE);
        this.globalBatchSize = (int) paramMap.get(HasGlobalBatchSize.GLOBAL_BATCH_SIZE);
        this.tol = (double) paramMap.get(HasTol.TOL);
        this.reg = (double) paramMap.get(HasReg.REG);
        this.elasticNet = (double) paramMap.get(HasElasticNet.ELASTIC_NET);
        this.method = method;
        this.lossFunc = lossFunc;
    }

    public OptimizerComponent withInitModel(String name) {
        initModel = name;
        return this;
    }

    @Override
    public MLData apply(MLData mlData) {
        DataStream<?> modelData;
        if (method == Method.SGD) {
            Optimizer optimizer =
                    new SGD(maxIter, learningRate, globalBatchSize, tol, reg, elasticNet);
            modelData = optimizer.optimize(mlData.slice(fromName, initModel), lossFunc);
            mlData.add(toName, modelData);
            return mlData;
        }
        throw new RuntimeException("not support yet.");
    }

    /** Comments. */
    public enum Method {
        SGD,
        FTRL,
        LBFGS
    }
}
