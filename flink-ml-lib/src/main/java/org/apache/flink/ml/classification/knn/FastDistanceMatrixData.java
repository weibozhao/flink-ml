package org.apache.flink.ml.classification.knn;

import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.util.Preconditions;

import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.core.JsonProcessingException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Save the data for calculating distance fast. The FastDistanceMatrixData saves several dense
 * vectors in a single matrix. The vectors are organized in columns, which means each column is a
 * single vector. For example, vec1: 0,1,2, vec2: 3,4,5, vec3: 6,7,8, then the data in matrix is
 * organized as: vec1,vec2,vec3. And the data array in <code>vectors</code> is {0,1,2,3,4,5,6,7,8}.
 */
public class FastDistanceMatrixData implements Serializable {

    /**
     * Stores several dense vectors in columns. For example, if the vectorSize is n, and matrix
     * saves m vectors, then the number of rows of <code>vectors</code> is n and the number of cols
     * of <code>vectors</code> is m.
     */
    public final DenseMatrix vectors;
    /**
     * Save the extra info besides the vector. Each vector is related to one row. Thus, for
     * FastDistanceVectorData, the length of <code>rows</code> is one. And for
     * FastDistanceMatrixData, the length of <code>rows</code> is equal to the number of cols of
     * <code>matrix</code>. Besides, the order of the rows are the same with the vectors.
     */
    public final String[] ids;

    /**
     * Stores some extra info extracted from the vector. It's also organized in columns. For
     * example, if we want to save the L1 norm and L2 norm of the vector, then the two values are
     * viewed as a two-dimension label vector. We organize the norm vectors together to get the
     * <code>label</code>. If the number of cols of <code>vectors</code> is m, then in this case the
     * dimension of <code>label</code> is 2 * m.
     */
    public DenseMatrix label;

    public String[] getIds() {
        return ids;
    }

    /**
     * Constructor, initialize the vector data and extra info.
     *
     * @param vectors DenseMatrix which saves vectors in columns.
     * @param ids extra info besides the vector.
     */
    public FastDistanceMatrixData(DenseMatrix vectors, String[] ids) {
        this.ids = ids;
        Preconditions.checkNotNull(vectors, "DenseMatrix should not be null!");
        if (null != ids) {
            Preconditions.checkArgument(
                    vectors.numCols() == ids.length,
                    "The column number of DenseMatrix must be equal to the rows array length!");
        }
        this.vectors = vectors;
    }

    /**
     * serialization of FastDistanceMatrixData.
     *
     * @return json string.
     */
    @Override
    public String toString() {
        Map<String, Object> params = new HashMap<>(3);
        try {
            params.put(
                    "vectors_val",
                    ReadWriteUtils.OBJECT_MAPPER.writeValueAsString(vectors.getData()));
            params.put(
                    "vectors_m",
                    ReadWriteUtils.OBJECT_MAPPER.writeValueAsString(vectors.numRows()));
            params.put(
                    "vectors_n",
                    ReadWriteUtils.OBJECT_MAPPER.writeValueAsString(vectors.numCols()));
            params.put(
                    "label_val", ReadWriteUtils.OBJECT_MAPPER.writeValueAsString(label.getData()));
            params.put("label_m", ReadWriteUtils.OBJECT_MAPPER.writeValueAsString(label.numRows()));
            params.put("label_n", ReadWriteUtils.OBJECT_MAPPER.writeValueAsString(label.numCols()));
            params.put("rows", ReadWriteUtils.OBJECT_MAPPER.writeValueAsString(ids));
            return ReadWriteUtils.OBJECT_MAPPER.writeValueAsString(params);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("FastDistanceMatrixData toString err...");
        }
    }

    /**
     * deserialization of FastDistanceMatrixData.
     *
     * @param modelStr string of model serialization.
     * @return FastDistanceMatrixData
     */
    public static FastDistanceMatrixData fromString(String modelStr)
            throws JsonProcessingException {
        Map<String, Object> params =
                ReadWriteUtils.OBJECT_MAPPER.readValue(modelStr, HashMap.class);
        String[] rows =
                ReadWriteUtils.OBJECT_MAPPER.readValue((String) params.get("rows"), String[].class);
        int m = ReadWriteUtils.OBJECT_MAPPER.readValue((String) params.get("vectors_m"), int.class);
        int n = ReadWriteUtils.OBJECT_MAPPER.readValue((String) params.get("vectors_n"), int.class);
        double[] val =
                ReadWriteUtils.OBJECT_MAPPER.readValue(
                        (String) params.get("vectors_val"), double[].class);
        DenseMatrix vectors = new DenseMatrix(m, n, val);
        int lm = ReadWriteUtils.OBJECT_MAPPER.readValue((String) params.get("label_m"), int.class);
        int ln = ReadWriteUtils.OBJECT_MAPPER.readValue((String) params.get("label_n"), int.class);
        double[] lval =
                ReadWriteUtils.OBJECT_MAPPER.readValue(
                        (String) params.get("label_val"), double[].class);
        FastDistanceMatrixData matrixData = new FastDistanceMatrixData(vectors, rows);
        matrixData.label = new DenseMatrix(lm, ln, lval);
        return matrixData;
    }
}
