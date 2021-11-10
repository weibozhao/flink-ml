package org.apache.flink.ml.params.knn;

import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.params.shared.colname.HasPredictionCol;
import org.apache.flink.ml.params.shared.colname.HasPredictionDetailCol;

/** knn predict parameters. */
public interface KnnPredictParams<T>
        extends HasPredictionCol<T>,
                HasPredictionDetailCol<T>,
                HasReservedColsDefaultAsNull<T>,
                HasVectorColDefaultAsNull<T> {
    /**
     * @cn-name topK
     * @cn topK
     */
    Param<Integer> K = new IntParam("k", "k", 10, ParamValidators.gt(0));

    default Integer getK() {
        return get(K);
    }

    default T setK(Integer value) {
        return set(K, value);
    }
}
