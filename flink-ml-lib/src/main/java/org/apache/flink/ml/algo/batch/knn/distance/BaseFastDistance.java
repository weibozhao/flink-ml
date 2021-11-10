package org.apache.flink.ml.algo.batch.knn.distance;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.linalg.DenseMatrix;
import org.apache.flink.ml.common.linalg.DenseVector;
import org.apache.flink.ml.common.linalg.MatVecOp;
import org.apache.flink.ml.common.linalg.SparseVector;
import org.apache.flink.ml.common.linalg.Vector;
import org.apache.flink.ml.common.linalg.VectorUtil;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.flink.shaded.curator4.com.google.common.collect.Iterables;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

/**
 * FastDistance is used to accelerate the speed of calculating the distance. The two main points are
 * as below:
 *
 * <p>1. By pre-calculating some extra info of the vector, such as L1 norm or L2 norm.
 *
 * <p>2. By organizing several vectors in a single matrix to increate the cache hit rate.
 *
 * <p>The first point applies for both dense and sparse vectors. The second points is only useful
 * for dense vector as the indices length of sparse vector is not the same and only works when we
 * need to access the vectors in batch.
 */
public abstract class BaseFastDistance implements ContinuousDistance {
    private static final long serialVersionUID = -316200445975691392L;
    /** Maximum size of a matrix. */
    private static final int SIZE = 5 * 1024 * 1024;

    private static final int MAX_ROW_NUMBER = (int) Math.sqrt(200 * 1024 * 1024 / 8.0);

    /**
     * Prepare the FastDistanceData, the output could be a list of FastDistanceVectorData if the
     * vector is sparse. If the vectors are dense, the output is a list of FastDistanceMatrixData.
     * As the size of the each element in a dataset is limited, we can not union all the dense
     * vectors in a single matrix. We must seperate the vectors and store them in several matrices.
     *
     * @param rows input rows.
     * @param vectorIdx the index of the vector columns.
     * @param keepIdxs the indexes of columns who are kept.
     * @return a list of <code>FastDistanceData</code>
     */
    public List<BaseFastDistanceData> prepareMatrixData(
            Iterable<Row> rows, int vectorIdx, int... keepIdxs) {
        Iterable<Tuple2<Vector, Row>> newItearble =
                Iterables.transform(
                        rows,
                        (row) -> {
                            Vector vec = VectorUtil.parse(row.getField(vectorIdx).toString());
                            row = getRow(row, keepIdxs);
                            return Tuple2.of(vec, row);
                        });
        return prepareMatrixData(newItearble);
    }

    /**
     * Prepare the FastDistanceData. If the vector is dense, we organize <code>
     * rowNumber = SIZE / 8 / vectorSize</code> vectors into a matrix. If the number of remaining
     * vectors is n ,which is less than rowNumber, then the last matrix only contains n columns.
     *
     * <p>If the vector is sparse, we deal with the inputs row by row and returns a list of
     * FastDistanceVectorData.
     *
     * @param tuples support vector and row input.
     * @return a list of FastDistanceData.
     */
    public List<BaseFastDistanceData> prepareMatrixData(Iterable<Tuple2<Vector, Row>> tuples) {
        Iterator<Tuple2<Vector, Row>> iterator = tuples.iterator();
        Tuple2<Vector, Row> row = null;
        int vectorSize = 0;
        boolean isDense = false;
        if (iterator.hasNext()) {
            row = iterator.next();
            if (row.f0 instanceof DenseVector) {
                isDense = true;
            }
            vectorSize = row.f0.size();
        }
        if (isDense) {
            return prepareDenseMatrixData(row, iterator, vectorSize);
        } else {
            return prepareSparseMatrixData(row, iterator, vectorSize);
        }
    }

    List<BaseFastDistanceData> prepareSparseMatrixData(
            Tuple2<Vector, Row> tuple, Iterator<Tuple2<Vector, Row>> iterator, int vectorSize) {
        final int rowNumber = MAX_ROW_NUMBER;
        List<BaseFastDistanceData> res = new ArrayList<>();
        if (vectorSize != -1) {
            while (null != tuple) {
                int index = 0;
                List<Integer>[] indices = new List[vectorSize];
                List<Double>[] values = new List[vectorSize];
                Row[] rows = new Row[rowNumber];
                while (index < rowNumber && null != tuple) {
                    Preconditions.checkState(
                            tuple.f0 instanceof SparseVector,
                            "Inputs should be the same vector type!");
                    rows[index] = tuple.f1;
                    MatVecOp.appendVectorToSparseData(
                            indices, values, index++, (SparseVector) tuple.f0);
                    tuple = iterator.hasNext() ? iterator.next() : null;
                }
                BaseFastDistanceData data =
                        index == rowNumber
                                ? new FastDistanceSparseData(indices, values, rowNumber, rows)
                                : new FastDistanceSparseData(
                                        indices, values, index, Arrays.copyOf(rows, index));
                updateLabel(data);
                res.add(data);
            }
            return res;
        } else {
            while (null != tuple) {
                int index = 0;
                HashMap<Integer, Tuple2<List<Integer>, List<Double>>> indexHashMap =
                        new HashMap<>(1);
                Row[] rows = new Row[rowNumber];
                while (index < rowNumber && null != tuple) {
                    Preconditions.checkState(
                            tuple.f0 instanceof SparseVector,
                            "Inputs should be the same vector type!");
                    rows[index] = tuple.f1;
                    MatVecOp.appendVectorToSparseData(
                            indexHashMap, index++, (SparseVector) tuple.f0);
                    tuple = iterator.hasNext() ? iterator.next() : null;
                }
                BaseFastDistanceData data =
                        index == rowNumber
                                ? new FastDistanceSparseData(indexHashMap, rowNumber, rows)
                                : new FastDistanceSparseData(
                                        indexHashMap, index, Arrays.copyOf(rows, index));
                updateLabel(data);
                res.add(data);
            }
            return res;
        }
    }

    List<BaseFastDistanceData> prepareDenseMatrixData(
            Tuple2<Vector, Row> tuple, Iterator<Tuple2<Vector, Row>> iterator, int vectorSize) {
        final int rowNumber = Math.min(SIZE / 8 / vectorSize, MAX_ROW_NUMBER);
        List<BaseFastDistanceData> res = new ArrayList<>();
        while (null != tuple) {
            int index = 0;
            DenseMatrix matrix = new DenseMatrix(vectorSize, rowNumber);
            Row[] rows = new Row[rowNumber];
            while (index < rowNumber && null != tuple) {
                Preconditions.checkState(
                        tuple.f0 instanceof DenseVector, "Inputs should be the same vector type!");
                rows[index] = tuple.f1;
                MatVecOp.appendVectorToMatrix(matrix, false, index++, tuple.f0);
                tuple = iterator.hasNext() ? iterator.next() : null;
            }
            BaseFastDistanceData data =
                    index == rowNumber
                            ? new FastDistanceMatrixData(matrix, rows)
                            : new FastDistanceMatrixData(
                                    new DenseMatrix(
                                            vectorSize,
                                            index,
                                            Arrays.copyOf(matrix.getData(), index * vectorSize)),
                                    Arrays.copyOf(rows, index));
            updateLabel(data);
            res.add(data);
        }
        return res;
    }

    /**
     * Prepare the FastDistanceVectorData.
     *
     * @param tuple support vector and row input.
     * @return FastDistanceVectorData.
     */
    public FastDistanceVectorData prepareVectorData(Tuple2<Vector, Row> tuple) {
        FastDistanceVectorData data = new FastDistanceVectorData(tuple.f0, tuple.f1);
        updateLabel(data);
        return data;
    }

    /**
     * Calculate the distances between vectors in <code>left</code> and <code>right</code>.
     *
     * @param left FastDistanceData.
     * @param right FastDistanceData.
     * @return a new DenseMatrix.
     */
    public DenseMatrix calc(BaseFastDistanceData left, BaseFastDistanceData right) {
        return calc(left, right, null);
    }

    /**
     * Calculate the distances between vectors in <code>left</code> and <code>right</code>. The
     * operation is a Cartesian Product of left and right. The inputs fall into four types: 1. left
     * is FastDistanceVectorData, right is FastDistanceVectorData, the dimension of the result
     * matrix is 1 X 1.
     *
     * <p>2. left is FastDistanceVectorData, right is FastDistanceMatrixData which saves m vectors.
     * The dimension of the result matrix is m X 1.
     *
     * <p>3. left is FastDistanceMatrixData which saves n vectors, right is FastDistanceVectorData.
     * The dimension of the result matrix is 1 X n.
     *
     * <p>4. left is FastDistanceMatrixData which saves n vectors, right is FastDistanceMatrixData
     * which saves m vectors. the dimension of the result matrix is m X n.
     *
     * @param left FastDistanceData.
     * @param right FastDistanceData.
     * @param res if res is null or the dimension of res is not satisfied, a new DenseMatrix is
     *     created, otherwise, the given res is refilled.
     * @return the distances.
     */
    public DenseMatrix calc(
            BaseFastDistanceData left, BaseFastDistanceData right, DenseMatrix res) {
        if (left instanceof FastDistanceVectorData) {
            if (right instanceof FastDistanceVectorData) {
                FastDistanceVectorData leftData = (FastDistanceVectorData) left;
                FastDistanceVectorData rightData = (FastDistanceVectorData) right;
                double d = calc(leftData, rightData);
                if (null == res || res.numCols() != 1 || res.numRows() != 1) {
                    res = new DenseMatrix(1, 1, new double[] {d});
                }
                res.set(0, 0, d);
            } else if (right instanceof FastDistanceMatrixData) {
                FastDistanceMatrixData matrixData = (FastDistanceMatrixData) right;
                if (null == res
                        || res.numRows() != matrixData.vectors.numCols()
                        || res.numCols() != 1) {
                    res = new DenseMatrix(matrixData.vectors.numCols(), 1);
                }
                calc((FastDistanceVectorData) left, matrixData, res.getData());
            } else {
                FastDistanceSparseData sparseData = (FastDistanceSparseData) right;
                if (null == res || res.numRows() != sparseData.vectorNum || res.numCols() != 1) {
                    res = new DenseMatrix(sparseData.vectorNum, 1);
                }
                calc((FastDistanceVectorData) left, sparseData, res.getData());
            }
        } else if (left instanceof FastDistanceMatrixData) {
            if (right instanceof FastDistanceVectorData) {
                FastDistanceMatrixData matrixData = (FastDistanceMatrixData) left;
                if (null == res
                        || res.numRows() != 1
                        || res.numCols() != matrixData.vectors.numCols()) {
                    res = new DenseMatrix(1, matrixData.vectors.numCols());
                }
                calc((FastDistanceVectorData) right, matrixData, res.getData());
            } else if (right instanceof FastDistanceMatrixData) {
                FastDistanceMatrixData leftData = (FastDistanceMatrixData) left;
                FastDistanceMatrixData rightData = (FastDistanceMatrixData) right;

                if (null == res
                        || res.numRows() != rightData.vectors.numCols()
                        || res.numCols() != leftData.vectors.numCols()) {
                    res = new DenseMatrix(rightData.vectors.numCols(), leftData.vectors.numCols());
                }
                calc(leftData, rightData, res);
            } else {
                throw new RuntimeException(
                        "Not support multiple dense vector and sparse vector distance calculation");
            }
        } else {
            if (right instanceof FastDistanceVectorData) {
                FastDistanceSparseData matrixData = (FastDistanceSparseData) left;
                if (null == res || res.numRows() != 1 || res.numCols() != matrixData.vectorNum) {
                    res = new DenseMatrix(1, matrixData.vectorNum);
                }
                calc((FastDistanceVectorData) right, matrixData, res.getData());
            } else if (right instanceof FastDistanceSparseData) {
                FastDistanceSparseData leftData = (FastDistanceSparseData) left;
                FastDistanceSparseData rightData = (FastDistanceSparseData) right;

                if (null == res
                        || res.numRows() != rightData.vectorNum
                        || res.numCols() != leftData.vectorNum) {
                    res = new DenseMatrix(rightData.vectorNum, leftData.vectorNum);
                }
                calc(leftData, rightData, res.getData());
            } else {
                throw new RuntimeException(
                        "Not support multiple dense vector and sparse vector distance calculation");
            }
        }
        return res;
    }

    /**
     * get sub row with special indices.
     *
     * @param row input row.
     * @param keepIdxs indices kept.
     * @return sub row.
     */
    public static Row getRow(Row row, int... keepIdxs) {
        Row res = null;
        if (null != keepIdxs) {
            res = new Row(keepIdxs.length);
            for (int i = 0; i < keepIdxs.length; i++) {
                res.setField(i, row.getField(keepIdxs[i]));
            }
        }
        return res;
    }

    /**
     * When a instance of FastDistanceData is created or the data inside is updated, we must update
     * the label as well.
     *
     * @param data FastDistanceData.
     */
    public abstract void updateLabel(BaseFastDistanceData data);

    /**
     * Calculate the distance between the two vectors in left and right.
     *
     * @param left FastDistanceVectorData.
     * @param right FastDistanceVectorData.
     * @return the distance.
     */
    abstract double calc(FastDistanceVectorData left, FastDistanceVectorData right);

    /**
     * Calculate the distances between a vector in <code>vector</code> and several vectors in <code>
     * matrix</code>. The result is a double array.
     *
     * @param vector FastDistanceVectorData.
     * @param matrix FastDistanceMatrixData.
     * @param res distances.
     */
    abstract void calc(FastDistanceVectorData vector, FastDistanceMatrixData matrix, double[] res);

    /**
     * Calcualate the distances between m vectors in <code>left</code> and n vectors in <code>right
     * </code>. The result is a n X m dimension matrix.
     *
     * @param left FastDistanceMatrixData.
     * @param right FastDistanceMatrixData.
     * @param res distances.
     */
    abstract void calc(FastDistanceMatrixData left, FastDistanceMatrixData right, DenseMatrix res);

    /**
     * Calcualate the distances between m vectors in <code>left</code> and n vectors in <code>right
     * </code>. The result is a n X m dimension matrix.
     *
     * @param left FastDistanceVectorData.
     * @param right FastDistanceSparseData.
     * @param res distances.
     */
    abstract void calc(FastDistanceVectorData left, FastDistanceSparseData right, double[] res);

    /**
     * Calcualate the distances between m vectors in <code>left</code> and n vectors in <code>right
     * </code>. The result is a n X m dimension matrix.
     *
     * @param left FastDistanceSparseData.
     * @param right FastDistanceSparseData.
     * @param res distances.
     */
    abstract void calc(FastDistanceSparseData left, FastDistanceSparseData right, double[] res);
}
