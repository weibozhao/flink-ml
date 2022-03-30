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

import org.apache.flink.ml.clustering.kmeans.KMeans;
import org.apache.flink.ml.clustering.kmeans.KMeansModelParams;
import org.apache.flink.ml.common.param.HasFeaturesCol;
import org.apache.flink.ml.common.param.HasLabelCol;
import org.apache.flink.ml.common.param.HasMaxIter;
import org.apache.flink.ml.common.param.HasSeed;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Params of {@link Ftrl}.
 *
 * @param <T> The class type of this instance.
 */
public interface FtrlParams<T> extends
    HasLabelCol <T>,
    HasFeaturesCol <T> {

	Param<Integer> VECTOR_SIZE =
		new IntParam("vectorSize", "The size of vector.", -1, ParamValidators.gt(0));

	default Integer getVectorSize() {
		return get(VECTOR_SIZE);
	}

	default T setVectorSize(Integer value) {
		return set(VECTOR_SIZE, value);
	}
}
