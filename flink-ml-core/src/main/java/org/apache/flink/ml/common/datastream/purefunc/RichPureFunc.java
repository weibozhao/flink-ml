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

package org.apache.flink.ml.common.datastream.purefunc;

import org.apache.flink.annotation.Experimental;

import java.io.Serializable;

/**
 * A base interface for all rich pure functions. This class defines methods for the life cycle of
 * the functions, as well as methods to access the context in which the functions are executed.
 */
@Experimental
public interface RichPureFunc extends Serializable {
    void open() throws Exception;

    void close() throws Exception;

    default void reset() throws Exception {
        close();
        open();
    }

    PureFuncContext getContext();

    void setContext(PureFuncContext context);
}
