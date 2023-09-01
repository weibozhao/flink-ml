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

import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.sharedobjects.AbstractSharedObjectsStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.types.Row;

import org.apache.commons.math3.analysis.function.Sigmoid;

/** Get required data for evaluating on training data. */
public class GetLabelPredOperator extends AbstractSharedObjectsStreamOperator<Row>
        implements OneInputStreamOperator<Integer, Row> {

    private static final Sigmoid sigmoid = new Sigmoid();
    private final String labelCol;
    private final String probCol;
    private final int interval;

    public GetLabelPredOperator(String labelCol, String probCol, int interval) {
        super();
        this.labelCol = labelCol;
        this.probCol = probCol;
        this.interval = interval;
    }

    @Override
    public void processElement(StreamRecord<Integer> element) throws Exception {
        invoke(
                (getter, setter) -> {
                    if (!getter.get(SharedObjectsConstants.NEED_INIT_TREE)) {
                        return;
                    }
                    if (getter.get(SharedObjectsConstants.ALL_TREES).size() % interval != 0) {
                        return;
                    }
                    LOG.info("Start to dump labels and predictors from memory to DataStream.");
                    BinnedInstance[] instances = getter.get(SharedObjectsConstants.INSTANCES);
                    double[] pgh = getter.get(SharedObjectsConstants.PREDS_GRADS_HESSIANS);
                    if (0 == pgh.length) {
                        pgh = new double[instances.length * 3];
                    }
                    Row row = Row.withNames();
                    for (int i = 0; i < instances.length; i += 1) {
                        row.setField(labelCol, instances[i].label);
                        row.setField(probCol, sigmoid.value(pgh[3 * i]));
                        output.collect(new StreamRecord<>(row));
                    }
                });
    }
}
