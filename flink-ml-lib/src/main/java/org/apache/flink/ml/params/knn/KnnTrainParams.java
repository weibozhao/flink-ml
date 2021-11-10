package org.apache.flink.ml.params.knn;

import org.apache.flink.ml.param.WithParams;

/** knn train parameters. */
public interface KnnTrainParams<T>
        extends WithParams<T>,
                HasLabelCol<T>,
                HasFeatureColsDefaultAsNull<T>,
                HasVectorColDefaultAsNull<T>,
                HasKnnDistanceType<T> {}
