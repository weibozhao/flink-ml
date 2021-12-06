package org.apache.flink.ml.classification.knn;

import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.util.Preconditions;

import java.io.Serializable;

/** Save the data for calculating distance fast. The FastDistanceMatrixData */
public class FastDistanceVectorData implements Serializable, Cloneable {
    /** Stores the vector(sparse or dense). */
    final DenseVector vector;

    /**
     * Stores some extra info extracted from the vector. For example, if we want to save the L1 norm
     * and L2 norm of the vector, then the two values are viewed as a two-dimension label vector.
     */
    public DenseVector label;

    /**
     * Constructor, initialize the vector data and extra info.
     *
     * @param vec vector.
     */
    public FastDistanceVectorData(DenseVector vec) {
        Preconditions.checkNotNull(vec, "Vector should not be null!");
        this.vector = vec;
    }
}
