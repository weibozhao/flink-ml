package org.apache.flink.ml.classification.knn;

import org.apache.flink.ml.common.param.HasLabelCol;

/** knn parameters. */
public interface KnnParams<T> extends HasLabelCol<T>, KnnModelParams<T> {}
