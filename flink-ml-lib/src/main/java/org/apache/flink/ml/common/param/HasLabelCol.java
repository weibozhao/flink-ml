package org.apache.flink.ml.common.param;

import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;
import org.apache.flink.ml.param.WithParams;

/** Interface for the shared labelCol param. */
public interface HasLabelCol<T> extends WithParams<T> {
    /** Label column name. */
    Param<String> LABEL_COL =
            new StringParam(
                    "labelCol",
                    "Name of the label column in the input table",
                    "label",
                    ParamValidators.notNull());

    default String getLabelCol() {
        return get(LABEL_COL);
    }

    default T setLabelCol(String colName) {
        return set(LABEL_COL, colName);
    }
}
