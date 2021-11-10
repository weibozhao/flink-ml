package org.apache.flink.ml.params.knn;

import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;
import org.apache.flink.ml.param.WithParams;

/**
 * Param of the name of the label column in the input table.
 *
 * @param <T>
 */
public interface HasLabelCol<T> extends WithParams <T> {
    /**
     * @cn-name 标签列名
     * @cn 输入表中的标签列名
     */
    Param <String> LABEL_COL = new StringParam("labelCol",
        "Name of the label column in the input table", null, ParamValidators.notNull());

    default String getLabelCol() {
        return get(LABEL_COL);
    }

    default T setLabelCol(String colName) {
        return set(LABEL_COL, colName);
    }
}
