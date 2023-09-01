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

package org.apache.flink.ml.feature;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.feature.transform.TypeTransform;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.math.BigDecimal;
import java.sql.Timestamp;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/** Tests the {@link TypeTransform}. */
public class TypeTransformTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table testTable;

    private static final List<Row> testData =
            Arrays.asList(
                    Row.of(1, 1.0, "1.0"),
                    Row.of(null, 2.3, "3.0"),
                    Row.of(2, null, "3.0"),
                    Row.of(2, 3.0, null),
                    Row.of(1, 2.3, "ddsef"));

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        testTable = tEnv.fromDataStream(env.fromCollection(testData)).as("f0", "f1", "class");
    }

    Table getTable(Row r1, Row r2) {
        List<Row> inputData = Arrays.asList(r1, r2);
        return tEnv.fromDataStream(env.fromCollection(inputData)).as("f1", "f2");
    }

    @Test
    public void test() throws Exception {
        TypeTransform transform =
                new TypeTransform()
                        .setToLongCols("f0")
                        .setToDoubleCols("f1", "class")
                        .setKeepOldCols(true)
                        .setDefaultLongValue(1000L)
                        .setDefaultDoubleValue(1000.0);
        Table output = transform.transform(testTable)[0];
        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());

        List<Row> expectedRows =
                Arrays.asList(
                        Row.of(2, 3.0, null, 2L, 3.0, 1000.0),
                        Row.of(null, 2.3, "3.0", 1000L, 2.3, 3.0),
                        Row.of(1, 1.0, "1.0", 1L, 1.0, 1.0),
                        Row.of(1, 2.3, "ddsef", 1L, 2.3, 1000.0),
                        Row.of(2, null, "3.0", 2L, 1000.0, 3.0));
        compareResultCollections(
                expectedRows, collectedResult, Comparator.comparingInt(Row::hashCode));
    }

    @Test
    public void testIntToLong() throws Exception {
        Table inputTable = getTable(Row.of(1, Integer.MAX_VALUE), Row.of(3, null));
        TypeTransform transform =
                new TypeTransform()
                        .setToLongCols("f1", "f2")
                        .setKeepOldCols(false)
                        .setDefaultLongValue(1000L);
        Table output = transform.transform(inputTable)[0];
        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());

        List<Row> expectedRows = Arrays.asList(Row.of(3L, 1000L), Row.of(1L, 2147483647L));
        compareResultCollections(
                expectedRows, collectedResult, Comparator.comparingInt(Row::hashCode));
    }

    @Test
    public void testIntToFloat() throws Exception {
        Table inputTable = getTable(Row.of(1, Integer.MAX_VALUE), Row.of(3, null));
        TypeTransform transform =
                new TypeTransform()
                        .setToFloatCols("f1", "f2")
                        .setKeepOldCols(false)
                        .setDefaultFloatValue(1000.0F);
        Table output = transform.transform(inputTable)[0];
        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());

        List<Row> expectedRows = Arrays.asList(Row.of(1.0f, 2147483647.0f), Row.of(3.0f, 1000.0f));
        compareResultCollections(
                expectedRows, collectedResult, Comparator.comparingInt(Row::hashCode));
    }

    @Test
    public void testIntToDouble() throws Exception {
        Table inputTable = getTable(Row.of(1, Integer.MAX_VALUE), Row.of(3, null));
        TypeTransform transform =
                new TypeTransform()
                        .setToDoubleCols("f1", "f2")
                        .setKeepOldCols(false)
                        .setDefaultDoubleValue(1000.0);
        Table output = transform.transform(inputTable)[0];

        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        List<Row> expectedRows = Arrays.asList(Row.of(1.0, 2147483647.0), Row.of(3.0, 1000.0));
        compareResultCollections(
                expectedRows, collectedResult, Comparator.comparingInt(Row::hashCode));
    }

    @Test
    public void testIntToString() throws Exception {
        Table inputTable = getTable(Row.of(1, Integer.MAX_VALUE), Row.of(3, null));
        TypeTransform transform =
                new TypeTransform()
                        .setToStringCols("f1", "f2")
                        .setKeepOldCols(false)
                        .setDefaultStringValue("1000");
        Table output = transform.transform(inputTable)[0];

        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        List<Row> expectedRows = Arrays.asList(Row.of("1", "2147483647"), Row.of("3", "1000"));
        compareResultCollections(
                expectedRows, collectedResult, Comparator.comparingInt(Row::hashCode));
    }

    @Test
    public void testDecimalToLong() throws Exception {
        Table inputTable =
                getTable(
                        Row.of(new BigDecimal("1.0"), BigDecimal.ZERO),
                        Row.of(BigDecimal.ONE, null));
        TypeTransform transform =
                new TypeTransform()
                        .setToLongCols("f1", "f2")
                        .setKeepOldCols(false)
                        .setDefaultLongValue(1000L);
        Table output = transform.transform(inputTable)[0];
        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());

        List<Row> expectedRows = Arrays.asList(Row.of(1L, 0L), Row.of(1L, 1000L));
        compareResultCollections(
                expectedRows, collectedResult, Comparator.comparingInt(Row::hashCode));
    }

    @Test
    public void testDecimalToFloat() throws Exception {
        Table inputTable =
                getTable(
                        Row.of(new BigDecimal("1.0"), BigDecimal.ZERO),
                        Row.of(BigDecimal.ONE, null));
        TypeTransform transform =
                new TypeTransform()
                        .setToFloatCols("f1", "f2")
                        .setKeepOldCols(false)
                        .setDefaultFloatValue(1000.0f);
        Table output = transform.transform(inputTable)[0];
        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());

        List<Row> expectedRows = Arrays.asList(Row.of(1.0f, 0.0f), Row.of(1.0f, 1000.0f));
        compareResultCollections(
                expectedRows, collectedResult, Comparator.comparingInt(Row::hashCode));
    }

    @Test
    public void testDecimalToDouble() throws Exception {
        Table inputTable =
                getTable(
                        Row.of(new BigDecimal("1.0"), BigDecimal.ZERO),
                        Row.of(BigDecimal.ONE, null));
        TypeTransform transform =
                new TypeTransform()
                        .setToDoubleCols("f1", "f2")
                        .setKeepOldCols(false)
                        .setDefaultDoubleValue(1000.0);
        Table output = transform.transform(inputTable)[0];

        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        List<Row> expectedRows = Arrays.asList(Row.of(1.0, 0.0), Row.of(1.0, 1000.0));
        compareResultCollections(
                expectedRows, collectedResult, Comparator.comparingInt(Row::hashCode));
    }

    @Test
    public void testDecimalToString() throws Exception {
        Table inputTable =
                getTable(
                        Row.of(new BigDecimal("1.0"), BigDecimal.ZERO),
                        Row.of(BigDecimal.ONE, null));
        TypeTransform transform =
                new TypeTransform()
                        .setToStringCols("f1", "f2")
                        .setKeepOldCols(false)
                        .setDefaultStringValue("1000");
        Table output = transform.transform(inputTable)[0];

        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        List<Row> expectedRows =
                Arrays.asList(
                        Row.of("1.000000000000000000", "0.000000000000000000"),
                        Row.of("1.000000000000000000", "1000"));
        compareResultCollections(
                expectedRows, collectedResult, Comparator.comparingInt(Row::hashCode));
    }

    @Test
    public void testDecimalToInt() throws Exception {
        Table inputTable =
                getTable(
                        Row.of(new BigDecimal("1.0"), BigDecimal.ZERO),
                        Row.of(BigDecimal.ONE, null));
        TypeTransform transform =
                new TypeTransform()
                        .setToIntCols("f1", "f2")
                        .setKeepOldCols(false)
                        .setDefaultIntValue(1000);
        Table output = transform.transform(inputTable)[0];

        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        List<Row> expectedRows = Arrays.asList(Row.of(1, 0), Row.of(1, 1000));
        compareResultCollections(
                expectedRows, collectedResult, Comparator.comparingInt(Row::hashCode));
    }

    @Test
    public void testLongToFloat() throws Exception {
        Table inputTable = getTable(Row.of(1L, Long.MIN_VALUE), Row.of(3L, null));
        TypeTransform transform =
                new TypeTransform()
                        .setToFloatCols("f1", "f2")
                        .setKeepOldCols(false)
                        .setDefaultFloatValue(1000.0f);
        Table output = transform.transform(inputTable)[0];

        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        List<Row> expectedRows = Arrays.asList(Row.of(1.0f, -9.223372E18f), Row.of(3.0f, 1000.0f));
        compareResultCollections(
                expectedRows, collectedResult, Comparator.comparingInt(Row::hashCode));
    }

    @Test
    public void testLongToDouble() throws Exception {
        Table inputTable = getTable(Row.of(1L, Long.MIN_VALUE), Row.of(3L, null));
        TypeTransform transform =
                new TypeTransform()
                        .setToDoubleCols("f1", "f2")
                        .setKeepOldCols(false)
                        .setDefaultDoubleValue(1000.0);
        Table output = transform.transform(inputTable)[0];

        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        List<Row> expectedRows =
                Arrays.asList(Row.of(1.0d, -9.223372036854776E18), Row.of(3.0d, 1000.0d));
        compareResultCollections(
                expectedRows, collectedResult, Comparator.comparingInt(Row::hashCode));
    }

    @Test
    public void testFloatToDouble() throws Exception {
        Table inputTable = getTable(Row.of(1.0f, Float.MAX_VALUE), Row.of(2.0f, null));
        TypeTransform transform =
                new TypeTransform()
                        .setToDoubleCols("f1", "f2")
                        .setKeepOldCols(false)
                        .setDefaultDoubleValue(1000.0);
        Table output = transform.transform(inputTable)[0];
        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        List<Row> expectedRows =
                Arrays.asList(Row.of(1.0, 3.4028234663852886E38), Row.of(2.0, 1000.0));
        compareResultCollections(
                expectedRows, collectedResult, Comparator.comparingInt(Row::hashCode));
    }

    @Test
    public void testDateToString() throws Exception {
        List<Row> binomialTrainData =
                Collections.singletonList(Row.of(1.0, new Timestamp(1685937761202L)));
        Table inputTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                binomialTrainData,
                                new RowTypeInfo(
                                        new TypeInformation[] {Types.DOUBLE, Types.SQL_TIMESTAMP},
                                        new String[] {"f1", "f2"})));
        TypeTransform transform =
                new TypeTransform()
                        .setToStringCols("f1", "f2")
                        .setKeepOldCols(true)
                        .setDefaultStringValue("null");
        Table output = transform.transform(inputTable)[0];
        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        List<Row> expectedRows =
                Collections.singletonList(
                        Row.of(
                                1.0,
                                new Timestamp(1685937761202L),
                                "1.0",
                                "2023-06-05 20:02:41.202"));
        compareResultCollections(
                expectedRows, collectedResult, Comparator.comparingInt(Row::hashCode));
    }

    @Test
    public void testErrorCols() throws Exception {
        List<Row> binomialTrainData =
                Collections.singletonList(Row.of(1.0, new Timestamp(1685937761202L)));
        Table inputTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                binomialTrainData,
                                new RowTypeInfo(
                                        new TypeInformation[] {Types.DOUBLE, Types.SQL_TIMESTAMP},
                                        new String[] {"f1", "f2"})));
        try {
            TypeTransform transform =
                    new TypeTransform()
                            .setToStringCols("f1", "f3")
                            .setKeepOldCols(true)
                            .setDefaultStringValue("null");
            Table output = transform.transform(inputTable)[0];
            IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        } catch (Exception e) {
            Assert.assertEquals(e.getMessage(), "Column: f3 doesn't exist in the input table.");
        }
    }
}
