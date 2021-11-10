package org.apache.flink.ml.algo.batch.knn.distance;

import org.apache.flink.ml.common.linalg.BLAS;
import org.apache.flink.ml.common.linalg.DenseMatrix;
import org.apache.flink.ml.common.linalg.DenseVector;
import org.apache.flink.ml.common.linalg.MatVecOp;
import org.apache.flink.ml.common.linalg.SparseVector;
import org.apache.flink.ml.common.linalg.Vector;
import org.apache.flink.statefun.flink.core.common.SerializableFunction;
import org.apache.flink.util.Preconditions;

import java.util.Arrays;

/**
 * Cosine calc is a measure of calc between two non-zero vectors of an inner product space that
 * measures the cosine of the angle between them.
 *
 * <p>https://en.wikipedia.org/wiki/Cosine_similarity
 *
 * <p>Given two vectors a and b, cosine calc = (a * b / ||a|| / ||b||), where a * b means dot
 * product, ||a|| means the L2 norm of the vector.
 *
 * <p>Cosine Distance is often used for the complement in positive space, cosine distance = 1 -
 * cosine calc.
 */
public class CosineDistance extends BaseFastDistance {
    private static final long serialVersionUID = 8579502553141821594L;
    /**
     * Calculate the Cosine distance between two arrays.
     *
     * @param array1 array1
     * @param array2 array2
     * @return the distance
     */
    @Override
    public double calc(double[] array1, double[] array2) {
        double dot = BLAS.dot(array1, array2);
        double cross = Math.sqrt(BLAS.dot(array1, array1) * BLAS.dot(array2, array2));
        return 1.0 - (cross > 0.0 ? dot / cross : 0.0);
    }

    /**
     * Calculate the distance between vec1 and vec2.
     *
     * @param vec1 vector1
     * @param vec2 vector2
     * @return the distance.
     */
    @Override
    public double calc(Vector vec1, Vector vec2) {
        double dot = MatVecOp.dot(vec1, vec2);
        double cross = vec1.normL2() * vec2.normL2();
        return 1.0 - (cross > 0.0 ? dot / cross : 0.0);
    }

    /**
     * For Cosine distance, distance = 1 - a * b / ||a|| / ||b|| So we can tranform the vector the
     * unit vector, and when we need to calculate the distance with another vector, only dot product
     * is calculated.
     *
     * @param data FastDistanceData.
     */
    @Override
    public void updateLabel(BaseFastDistanceData data) {
        if (data instanceof FastDistanceVectorData) {
            FastDistanceVectorData vectorData = (FastDistanceVectorData) data;
            double vectorLabel = Math.sqrt(MatVecOp.dot(vectorData.vector, vectorData.vector));
            if (vectorLabel > 0) {
                vectorData.vector.scaleEqual(1.0 / vectorLabel);
            }
        } else if (data instanceof FastDistanceMatrixData) {
            FastDistanceMatrixData matrix = (FastDistanceMatrixData) data;
            int vectorSize = matrix.vectors.numRows();
            double[] matrixData = matrix.vectors.getData();

            int cnt = 0;
            while (cnt < matrixData.length) {
                int endIndex = cnt + vectorSize;
                double vectorLabel = 0.;
                while (cnt < endIndex) {
                    vectorLabel += matrixData[cnt] * matrixData[cnt];
                    cnt++;
                }
                vectorLabel = Math.sqrt(vectorLabel);
                if (vectorLabel > 0) {
                    BLAS.scal(1.0 / vectorLabel, matrixData, cnt - vectorSize, vectorSize);
                }
            }
        } else {
            FastDistanceSparseData sparseData = (FastDistanceSparseData) data;
            double[] vectorLabel = new double[sparseData.vectorNum];
            int[][] indices = sparseData.getIndices();
            double[][] values = sparseData.getValues();
            for (int i = 0; i < indices.length; i++) {
                if (null != indices[i]) {
                    for (int j = 0; j < indices[i].length; j++) {
                        vectorLabel[indices[i][j]] += (values[i][j] * values[i][j]);
                    }
                }
            }
            for (int i = 0; i < vectorLabel.length; i++) {
                vectorLabel[i] = Math.sqrt(vectorLabel[i]);
            }
            for (int i = 0; i < indices.length; i++) {
                if (null != indices[i]) {
                    for (int j = 0; j < indices[i].length; j++) {
                        values[i][j] /= vectorLabel[indices[i][j]];
                    }
                }
            }
        }
    }

    /**
     * distance = 1 - a * b / ||a|| / ||b|
     *
     * @param left single vector with label(reciprocal of L2 norm)
     * @param right single vector with label(reciprocal of L2 norm)
     * @return the distance
     */
    @Override
    double calc(FastDistanceVectorData left, FastDistanceVectorData right) {
        return 1.0 - MatVecOp.dot(left.vector, right.vector);
    }

    /**
     * distance = 1 - a * b / ||a|| / ||b|
     *
     * @param leftVector single vector with label(reciprocal of L2 norm)
     * @param rightVectors vectors with labels(reciprocal of L2 norm array)
     * @param res the distances between leftVector and all the vectors in rightVectors.
     */
    @Override
    void calc(
            FastDistanceVectorData leftVector, FastDistanceMatrixData rightVectors, double[] res) {
        baseCalc(leftVector, rightVectors, res);
    }

    static void baseCalc(
            FastDistanceVectorData leftVector, FastDistanceMatrixData rightVectors, double[] res) {
        Arrays.fill(res, 1);
        BLAS.gemv(-1, rightVectors.vectors, true, leftVector.vector, 1.0, new DenseVector(res));
    }

    /**
     * distance = 1 - a * b / ||a|| / ||b|
     *
     * @param left vectors with labels(reciprocal of L2 norm array)
     * @param right vectors with labels(reciprocal of L2 norm array)
     * @param res the distances between all the vectors in left and all the vectors in right.
     */
    @Override
    void calc(FastDistanceMatrixData left, FastDistanceMatrixData right, DenseMatrix res) {
        Arrays.fill(res.getData(), 1);
        BLAS.gemm(-1, right.vectors, true, left.vectors, false, 1.0, res);
    }

    @Override
    void calc(FastDistanceVectorData left, FastDistanceSparseData right, double[] res) {
        baseCalc(left, right, res, x -> -x);
    }

    static void baseCalc(
            FastDistanceVectorData left,
            FastDistanceSparseData right,
            double[] res,
            SerializableFunction<Double, Double> function) {
        Arrays.fill(res, 1.0);
        int[][] rightIndices = right.getIndices();
        double[][] rightValues = right.getValues();

        if (left.vector instanceof DenseVector) {
            double[] vector = ((DenseVector) left.vector).getData();
            for (int i = 0; i < vector.length; i++) {
                if (null != rightIndices[i]) {
                    for (int j = 0; j < rightIndices[i].length; j++) {
                        res[rightIndices[i][j]] -= rightValues[i][j] * vector[i];
                    }
                }
            }
        } else {
            SparseVector vector = (SparseVector) left.getVector();
            int[] indices = vector.getIndices();
            double[] values = vector.getValues();
            for (int i = 0; i < indices.length; i++) {
                if (null != rightIndices[indices[i]]) {
                    for (int j = 0; j < rightIndices[indices[i]].length; j++) {
                        res[rightIndices[indices[i]][j]] +=
                                function.apply(rightValues[indices[i]][j] * values[i]);
                    }
                }
            }
        }
    }

    @Override
    void calc(FastDistanceSparseData left, FastDistanceSparseData right, double[] res) {
        baseCalc(left, right, res, x -> -x);
    }

    static void baseCalc(
            FastDistanceSparseData left,
            FastDistanceSparseData right,
            double[] res,
            SerializableFunction<Double, Double> function) {
        Arrays.fill(res, 1.0);
        int[][] leftIndices = left.getIndices();
        int[][] rightIndices = right.getIndices();
        double[][] leftValues = left.getValues();
        double[][] rightValues = right.getValues();

        Preconditions.checkArgument(
                leftIndices.length == rightIndices.length, "VectorSize not equal!");
        for (int i = 0; i < leftIndices.length; i++) {
            int[] leftIndicesList = leftIndices[i];
            int[] rightIndicesList = rightIndices[i];
            double[] leftValuesList = leftValues[i];
            double[] rightValuesList = rightValues[i];
            if (null != leftIndicesList) {
                for (int j = 0; j < leftIndicesList.length; j++) {
                    double leftValue = leftValuesList[j];
                    int startIndex = leftIndicesList[j] * right.vectorNum;
                    if (null != rightIndicesList) {
                        for (int k = 0; k < rightIndicesList.length; k++) {
                            res[startIndex + rightIndicesList[k]] +=
                                    function.apply(rightValuesList[k] * leftValue);
                        }
                    }
                }
            }
        }
    }
}
