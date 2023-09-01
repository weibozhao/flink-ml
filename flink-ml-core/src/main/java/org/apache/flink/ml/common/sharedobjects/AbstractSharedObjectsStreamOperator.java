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

package org.apache.flink.ml.common.sharedobjects;

import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.util.function.BiConsumerWithException;

import java.util.UUID;

/**
 * A default implementation of {@link AbstractStreamOperator} which implements {@link
 * SharedObjectsStreamOperator}. Use this class to reduce boilerplate codes.
 */
public abstract class AbstractSharedObjectsStreamOperator<OUT> extends AbstractStreamOperator<OUT>
        implements SharedObjectsStreamOperator {

    private final String sharedObjectsAccessorID;
    private transient SharedObjectsContext sharedObjectsContext;

    protected AbstractSharedObjectsStreamOperator() {
        super();
        sharedObjectsAccessorID = getClass().getSimpleName() + "-" + UUID.randomUUID();
    }

    public void onSharedObjectsContextSet(SharedObjectsContext context) {
        sharedObjectsContext = context;
    }

    public String getSharedObjectsAccessorID() {
        return sharedObjectsAccessorID;
    }

    public void invoke(
            BiConsumerWithException<
                            SharedObjectsContext.SharedItemGetter,
                            SharedObjectsContext.SharedItemSetter,
                            Exception>
                    func)
            throws Exception {
        sharedObjectsContext.invoke(func);
    }
}
