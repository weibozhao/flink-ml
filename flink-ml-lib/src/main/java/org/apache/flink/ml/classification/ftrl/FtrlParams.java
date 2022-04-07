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

package org.apache.flink.ml.classification.ftrl;

import org.apache.flink.ml.common.param.HasBatchStrategy;
import org.apache.flink.ml.common.param.HasFeaturesCol;
import org.apache.flink.ml.common.param.HasGlobalBatchSize;
import org.apache.flink.ml.common.param.HasLabelCol;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;

/**
 * Params of {@link FtrlSplitVec}.
 *
 * @param <T> The class type of this instance.
 */
public interface FtrlParams<T> extends
    HasLabelCol <T>,
	HasBatchStrategy <T>,
	HasGlobalBatchSize <T>,
    HasFeaturesCol <T> {

	Param<Integer> VECTOR_SIZE =
		new IntParam("vectorSize", "The size of vector.", -1, ParamValidators.gt(-2));

	default Integer getVectorSize() {
		return get(VECTOR_SIZE);
	}

	default T setVectorSize(Integer value) {
		return set(VECTOR_SIZE, value);
	}

	Param<Double> L_1 =
		new DoubleParam("l1", "The parameter l1 of ftrl.", 0.1, ParamValidators.gt(0.0));

	default Double getL1() {
		return get(L_1);
	}

	default T setL1(Double value) {
		return set(L_1, value);
	}

	Param<Double> L_2 =
		new DoubleParam("l2", "The parameter l2 of ftrl.", 0.1, ParamValidators.gt(0.0));

	default Double getL2() {
		return get(L_2);
	}

	default T setL2(Double value) {
		return set(L_2, value);
	}

	Param<Double> ALPHA =
		new DoubleParam("alpha", "The parameter alpha of ftrl.", 0.1, ParamValidators.gt(0.0));

	default Double getAlpha() {
		return get(ALPHA);
	}

	default T setAlpha(Double value) {
		return set(ALPHA, value);
	}

	Param<Double> BETA =
		new DoubleParam("alpha", "The parameter beta of ftrl.", 0.1, ParamValidators.gt(0.0));

	default Double getBETA() {
		return get(BETA);
	}

	default T setBETA(Double value) {
		return set(BETA, value);
	}
}
