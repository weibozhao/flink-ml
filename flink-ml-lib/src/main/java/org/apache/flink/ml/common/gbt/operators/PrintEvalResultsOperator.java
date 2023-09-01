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

package org.apache.flink.ml.common.gbt.operators;

import org.apache.flink.ml.common.sharedobjects.AbstractSharedObjectsStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.types.Row;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Print evaluation results. */
public class PrintEvalResultsOperator extends AbstractSharedObjectsStreamOperator<Row>
        implements OneInputStreamOperator<Row, Row> {

    private static final Logger LOG = LoggerFactory.getLogger(PrintEvalResultsOperator.class);

    @Override
    public void processElement(StreamRecord<Row> element) throws Exception {
        invoke(
                (getter, setter) -> {
                    int numTrees = getter.get(SharedObjectsConstants.ALL_TREES).size();
                    double auc = element.getValue().<Double>getFieldAs(0);
                    LOG.info("AUC for training set after {}-th tree trained: {}", numTrees, auc);
                    System.err.printf(
                            "AUC for training set after %d-th tree trained: %f%n", numTrees, auc);
                });
    }
}
