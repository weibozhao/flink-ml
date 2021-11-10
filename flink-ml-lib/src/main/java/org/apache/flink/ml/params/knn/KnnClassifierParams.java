package org.apache.flink.ml.params.knn;

import org.apache.flink.ml.param.WithParams;

/**
 * knn train parameters.
 */
public interface KnnClassifierParams<T>
	extends KnnTrainParams<T>, KnnPredictParams<T> {}
