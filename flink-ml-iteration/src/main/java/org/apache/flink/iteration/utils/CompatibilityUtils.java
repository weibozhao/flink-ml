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

package org.apache.flink.iteration.utils;

import org.apache.flink.annotation.Internal;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.core.memory.ManagedMemoryUseCase;
import org.apache.flink.streaming.api.graph.StreamConfig;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Utility methods to maintain compatibility between community Flink and VVR. */
@Internal
public class CompatibilityUtils {

    private static final Logger LOG = LoggerFactory.getLogger(CompatibilityUtils.class);

    /**
     * Gets managed memory fraction for state backend.
     *
     * <p>With VVR version, `getManagedMemoryFractionForStateBackend` should be used, while with
     * community version, `getManagedMemoryFractionOperatorUseCaseOfSlot` is used. Since there is no
     * method `getManagedMemoryFractionForStateBackend` in the community version, we use try-catch
     * to implement above logic.
     */
    public static double getManagedMemoryFractionForStateBackend(
            StreamConfig streamConfig, Configuration taskManagerConfig, ClassLoader cl) {
        try {
            return streamConfig.getManagedMemoryFractionForStateBackend(taskManagerConfig, cl);
        } catch (NoSuchMethodError ignored) {
            LOG.info(
                    "Call getManagedMemoryFractionForStateBackend failed, use getManagedMemoryFractionOperatorUseCaseOfSlot instead.");
            return streamConfig.getManagedMemoryFractionOperatorUseCaseOfSlot(
                    ManagedMemoryUseCase.STATE_BACKEND, taskManagerConfig, cl);
        }
    }
}
