package org.apache.flink.ml.params.knn;

import org.apache.flink.ml.algo.batch.knn.distance.CosineDistance;
import org.apache.flink.ml.algo.batch.knn.distance.EuclideanDistance;
import org.apache.flink.ml.algo.batch.knn.distance.BaseFastDistance;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.WithParams;
import org.apache.flink.ml.params.ParamUtil;

import java.io.Serializable;

/** Params: Distance type for clustering, support EUCLIDEAN and COSINE. */
public interface HasKnnDistanceType<T> extends WithParams <T> {
    /**
     * @cn-name 距离度量方式
     * @cn 聚类使用的距离类型
     */
    Param <DistanceType> DISTANCE_TYPE = new Param <>("distanceType", DistanceType.class,
        "Distance type for clustering", DistanceType.EUCLIDEAN, ParamValidators.notNull());

    default DistanceType getDistanceType() {
        return get(DISTANCE_TYPE);
    }

    default T setDistanceType(DistanceType value) {
        return set(DISTANCE_TYPE, value);
    }

    default T setDistanceType(String value) {
        return set(DISTANCE_TYPE, ParamUtil.searchEnum(DISTANCE_TYPE, value));
    }

    /** Various distance types. */
    enum DistanceType implements Serializable {
        /** EUCLIDEAN */
        EUCLIDEAN(new EuclideanDistance()),
        /** COSINE */
        COSINE(new CosineDistance());

        public BaseFastDistance getFastDistance() {
            return fastDistance;
        }

        private BaseFastDistance fastDistance;

        DistanceType(BaseFastDistance fastDistance) {
            this.fastDistance = fastDistance;
        }
    }
}
