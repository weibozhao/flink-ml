package org.apache.flink.ml.params.knn;

import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.WithParams;

/**
 * Params of the names of the feature columns used for training in the input table.
 */
public interface HasFeatureColsDefaultAsNull<T> extends WithParams <T> {
	/**
	 * @cn-name 特征列名数组
	 * @cn 特征列名数组，默认全选
	 */
	Param <String[]> FEATURE_COLS = new Param <>("featureCols", String[].class,
		"Names of the feature columns used for training in the input table", null, ParamValidators.alwaysTrue());

	default String[] getFeatureCols() {
		return get(FEATURE_COLS);
	}

	default T setFeatureCols(String... colNames) {
		return set(FEATURE_COLS, colNames);
	}
}
