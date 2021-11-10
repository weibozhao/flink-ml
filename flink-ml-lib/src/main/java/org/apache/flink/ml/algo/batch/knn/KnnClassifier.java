package org.apache.flink.ml.algo.batch.knn;

import org.apache.flink.ml.api.core.Estimator;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.params.knn.KnnClassifierParams;
import org.apache.flink.table.api.Table;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * KNN classifier is to classify unlabeled observations by assigning them to the class of the most
 * similar labeled examples.
 */
public class KnnClassifier
        implements Estimator<KnnClassifier, KnnClassificationModel>,
                KnnClassifierParams<KnnClassifier> {

    private static final long serialVersionUID = 5292477422193301398L;
    protected Map<Param<?>, Object> params;
    /** constructor. */
    public KnnClassifier() {
        super();
    }

    /**
     * constructor.
     *
     * @param params parameters for algorithm.
     */
    public KnnClassifier(Map<Param<?>, Object> params) {
        this.params = params;
    }

    /**
     * @param inputs a list of tables
     * @return knn classification model.
     */
    @Override
    public KnnClassificationModel fit(Table... inputs) {
        Table[] model = new KnnTrainBatchOp(params).transform(inputs);
        KnnClassificationModel knnClassificationModel = new KnnClassificationModel(params);
        knnClassificationModel.setModelData(model[0]);
        return knnClassificationModel;
    }

    /** @return parameters for algorithm. */
    @Override
    public Map<Param<?>, Object> getParamMap() {
        if (null == this.params) {
            this.params = new HashMap<>(1);
        }
        return this.params;
    }

    @Override
    public void save(String path) throws IOException {}
}
