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
import org.apache.flink.ml.param.WithParams;

/**
 * An interface for classes with a parameter specifying the name of multiple table columns with null
 * default value.
 *
 * @see HasSelectedCol
 * @see HasSelectedColDefaultAsNull
 * @see HasSelectedCols
 */
public interface HasSelectedColsDefaultAsNull<T> extends WithParams<T> {

    Param<String[]> SELECTED_COLS =
            new Param<>(
                    "selectedCols",
                    String[].class,
                    "Names of the columns used for processing",
                    null,
                    null);

    default String[] getSelectedCols() {
        return get(SELECTED_COLS);
    }

    default T setSelectedCols(String... value) {
        return set(SELECTED_COLS, value);
    }
}
