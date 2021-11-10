package org.apache.flink.ml.algo.batch.knn;

import org.apache.flink.ml.api.core.Model;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.params.knn.KnnPredictParams;
import org.apache.flink.table.api.Table;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/** Knn classification model fitted by KnnClassifier. */
public class KnnClassificationModel
        implements Model <KnnClassificationModel>, KnnPredictParams<KnnClassificationModel> {

    private static final long serialVersionUID = 1303892137143865652L;
    protected Map <Param <?>, Object> params;

    private Table modelData;

    /** constructor. */
    public KnnClassificationModel() {
        this(null);
    }

    /**
     * constructor.
     *
     * @param params parameters for algorithm.
     */
    public KnnClassificationModel(Map<Param <?>, Object> params) {
        this.params = params;
    }

    /**
     * Set model data for knn prediction.
     *
     * @param modelData knn model.
     * @return knn classification model.
     */
    public void setModelData(Table modelData) {
        this.modelData = modelData;
    }

    /**
     * @param inputs a list of tables.
     * @return result.
     */
    @Override
    public Table[] transform(Table... inputs) {
        return new KnnPredictBatchOp(this.params).transform(inputs[0], modelData);
    }

    /**
     *
     * @return parameters for algorithm.
     */
    @Override
    public Map <Param <?>, Object> getParamMap() {
        if (null == this.params) {
            this.params = new HashMap <>(1);
        }
        return this.params;
    }

    @Override
    public void save(String path) throws IOException {}
}
