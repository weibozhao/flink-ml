package org.apache.flink.ml.params.knn;

import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.StringParam;
import org.apache.flink.ml.param.WithParams;

/** Trait for parameter vectorColName. */
public interface HasVectorColDefaultAsNull<T> extends WithParams<T> {

    /**
     * @cn-name 向量列名
     * @cn 向量列对应的列名，默认值是null
     */
    Param<String> VECTOR_COL = new StringParam("vectorCol", "Name of a vector column", null);

    default String getVectorCol() {
        return get(VECTOR_COL);
    }

    default T setVectorCol(String colName) {
        return set(VECTOR_COL, colName);
    }
}
