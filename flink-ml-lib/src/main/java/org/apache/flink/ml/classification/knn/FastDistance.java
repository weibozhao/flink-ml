package org.apache.flink.ml.classification.knn;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.flink.shaded.curator4.com.google.common.collect.Iterables;

import dev.ludovic.netlib.blas.F2jBLAS;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * FastDistance is an accelerated distance calculating method. It use matrix vector operation to
 * improve speed of distance calculating.
 *
 * <p>The distance type in this class is euclidean distance:
 *
 * <p>https://en.wikipedia.org/wiki/Euclidean_distance
 */
public class FastDistance implements Serializable {
    /** Label size. */
    private static final int LABEL_SIZE = 1;

    /** Maximum size of a matrix. */
    private static final int SIZE = 5 * 1024 * 1024;

    private static final int MAX_ROW_NUMBER = (int) Math.sqrt(200 * 1024 * 1024 / 8.0);

    /** The blas used to accelerating speed. */
    private static final dev.ludovic.netlib.blas.F2jBLAS NATIVE_BLAS =
            (F2jBLAS) F2jBLAS.getInstance();

    /**
     * Prepare the FastDistanceData, the output is a list of FastDistanceMatrixData. As the size of
     * the each element in a dataset is limited, we can not union all the dense vectors in a single
     * matrix. We must seperate the vectors and store them in several matrices.
     *
     * @param rows input rows.
     * @param vectorIdx the index of the vector columns.
     * @param keepIdxs the indexes of columns who are kept.
     * @return a list of <code>FastDistanceData</code>
     */
    public List<FastDistanceMatrixData> prepareMatrixData(
            Iterable<Row> rows, int vectorIdx, int... keepIdxs) {
        Iterable<Tuple2<DenseVector, Row>> newItearble =
                Iterables.transform(
                        rows,
                        (row) -> {
                            DenseVector vec = (DenseVector) row.getField(vectorIdx);
                            Row ret = (null != keepIdxs) ? null : row;
                            if (null != keepIdxs) {
                                ret = new Row(keepIdxs.length);
                                for (int i = 0; i < keepIdxs.length; i++) {
                                    ret.setField(i, row.getField(keepIdxs[i]));
                                }
                            }
                            return Tuple2.of(vec, ret);
                        });

        Iterator<Tuple2<DenseVector, Row>> iterator = newItearble.iterator();
        Tuple2<DenseVector, Row> row = iterator.next();
        int vectorSize = row.f0.size();
        final int rowNumber = Math.min(SIZE / 8 / vectorSize, MAX_ROW_NUMBER);
        List<FastDistanceMatrixData> res = new ArrayList<>();
        while (null != row) {
            int index = 0;
            DenseMatrix matrix = new DenseMatrix(vectorSize, rowNumber);
            String[] ids = new String[rowNumber];
            while (index < rowNumber && null != row) {
                Preconditions.checkState(
                        row.f0 instanceof DenseVector, "Inputs should be the same vector type!");
                ids[index] = row.f1.getField(0).toString();
                double[] vectorData = row.f0.toArray();
                double[] matrixData = matrix.getData();
                int vecSize = vectorData.length;
                System.arraycopy(vectorData, 0, matrixData, index * vecSize, vecSize);
                index++;

                row = iterator.hasNext() ? iterator.next() : null;
            }
            FastDistanceMatrixData data =
                    index == rowNumber
                            ? new FastDistanceMatrixData(matrix, ids)
                            : new FastDistanceMatrixData(
                                    new DenseMatrix(
                                            vectorSize,
                                            index,
                                            Arrays.copyOf(matrix.getData(), index * vectorSize)),
                                    Arrays.copyOf(ids, index));
            updateMatrixLabel(data);
            res.add(data);
        }
        return res;
    }

    /**
     * Prepare the FastDistanceVectorData.
     *
     * <p>For Euclidean distance, distance = sqrt((a - b)^2) = (sqrt(a^2 + b^2 - 2ab)) So we can
     * pre-calculate the L2 norm square of the vector, and when we need to calculate the distance
     * with another vector, only dot product is calculated. For FastDistanceVectorData, the label is
     * a one-dimension vector.
     *
     * @param tuple support vector and row input.
     * @return FastDistanceVectorData.
     */
    public FastDistanceVectorData prepareVectorData(Tuple2<DenseVector, Row> tuple) {
        FastDistanceVectorData vectorData = new FastDistanceVectorData(tuple.f0);
        double d = 0.0;
        for (int i = 0; i < vectorData.vector.size(); ++i) {
            d += vectorData.vector.values[i] * vectorData.vector.values[i];
        }
        if (vectorData.label == null || vectorData.label.size() != LABEL_SIZE) {
            vectorData.label = new DenseVector(new double[LABEL_SIZE]);
        }
        vectorData.label.values[0] = d;
        return vectorData;
    }

    /**
     * Calculate the distances between vectors in <code>left</code> and <code>right</code>.
     *
     * @param left FastDistanceVectorData.
     * @param right FastDistanceMatrixData.
     * @return a new DenseMatrix.
     */
    public DenseMatrix calc(FastDistanceVectorData left, FastDistanceMatrixData right) {
        DenseMatrix res = new DenseMatrix(right.vectors.numCols(), 1);
        double[] normL2Square = right.label.getData();

        final int m = right.vectors.numRows();
        final int n = right.vectors.numCols();
        final int lda = right.vectors.numRows();
        NATIVE_BLAS.dgemv(
                "T",
                m,
                n,
                -2.0,
                right.vectors.getData(),
                lda,
                left.vector.toArray(),
                1,
                0.0,
                res.getData(),
                1);

        double vecLabel = left.label.values[0];
        for (int i = 0; i < res.getData().length; i++) {
            res.getData()[i] = Math.sqrt(Math.abs(res.getData()[i] + vecLabel + normL2Square[i]));
        }
        return res;
    }

    /**
     * For Euclidean distance, distance = sqrt((a - b)^2) = (sqrt(a^2 + b^2 - 2ab)) So we can
     * pre-calculate the L2 norm square of the vector, and when we need to calculate the distance
     * with another vector, only dot product is calculated. For FastDistanceVectorData, the label is
     * a one-dimension vector. For FastDistanceMatrixData, the label is a 1 X n DenseMatrix, n is
     * the number of vectors saved in the matrix.
     *
     * @param matrix FastDistanceMatrixData.
     */
    private void updateMatrixLabel(FastDistanceMatrixData matrix) {
        int vectorSize = matrix.vectors.numRows();
        int numVectors = matrix.vectors.numCols();
        if (matrix.label == null
                || matrix.label.numCols() != numVectors
                || matrix.label.numRows() != LABEL_SIZE) {
            matrix.label = new DenseMatrix(LABEL_SIZE, numVectors);
        }
        double[] label = matrix.label.getData();
        double[] matrixData = matrix.vectors.getData();
        Arrays.fill(label, 0.0);
        int labelCnt = 0;
        int cnt = 0;
        while (cnt < matrixData.length) {
            int endIndex = cnt + vectorSize;
            while (cnt < endIndex) {
                label[labelCnt] += matrixData[cnt] * matrixData[cnt];
                cnt++;
            }
            labelCnt++;
        }
    }
}
