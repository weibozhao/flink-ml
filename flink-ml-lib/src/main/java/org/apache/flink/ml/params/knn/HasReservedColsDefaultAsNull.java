package org.apache.flink.ml.params.knn;

import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.WithParams;

/**
 * An interface for classes with a parameter specifying the names of the columns to be retained in
 * the output table.
 */
public interface HasReservedColsDefaultAsNull<T> extends WithParams <T> {
    /**
     * @cn-name 算法保留列名
     * @cn 算法保留列
     */
    Param <String[]> RESERVED_COLS =new Param <>("reservedCols", String[].class,
        "Names of the columns to be retained in the output table", null, ParamValidators.alwaysTrue());

    default String[] getReservedCols() {
        return get(RESERVED_COLS);
    }

    default T setReservedCols(String... colNames) {
        return set(RESERVED_COLS, colNames);
    }
}
