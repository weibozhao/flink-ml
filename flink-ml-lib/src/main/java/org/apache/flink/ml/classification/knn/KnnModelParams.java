package org.apache.flink.ml.classification.knn;

import org.apache.flink.ml.common.param.HasFeaturesCol;
import org.apache.flink.ml.common.param.HasK;
import org.apache.flink.ml.common.param.HasPredictionCol;

/** knn model parameters. */
public interface KnnModelParams<T> extends HasFeaturesCol<T>, HasPredictionCol<T>, HasK<T> {}
