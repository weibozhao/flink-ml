/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.flink.ml.params.shared.colname;

import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.StringParam;
import org.apache.flink.ml.param.WithParams;
import org.apache.flink.ml.params.knn.HasOutputCol;

/**
 * An interface for classes with a parameter specifying name of the output column with a null
 * default value.
 *
 * @see HasOutputCol
 * @see HasOutputCols
 * @see HasOutputColsDefaultAsNull
 */
public interface HasOutputColDefaultAsNull<T> extends WithParams<T> {

    Param<String> OUTPUT_COL =
            new StringParam("outputCol", "Name of the output column", null, null);

    default String getOutputCol() {
        return get(OUTPUT_COL);
    }

    default T setOutputCol(String value) {
        return set(OUTPUT_COL, value);
    }
}
