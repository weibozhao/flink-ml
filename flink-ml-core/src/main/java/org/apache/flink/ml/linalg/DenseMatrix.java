package org.apache.flink.ml.linalg;

import java.io.Serializable;

/**
 * DenseMatrix stores dense matrix data and provides some methods to operate on the matrix it
 * represents. This data structure helps knn to accelerate distance calculation.
 */
public class DenseMatrix implements Serializable {

    /** Row dimension. */
    int m;

    /** Column dimension. */
    int n;

    /**
     * Array for internal storage of elements.
     *
     * <p>The matrix data is stored in column major format internally.
     */
    double[] data;

    /**
     * Construct an m-by-n matrix of zeros.
     *
     * @param m Number of rows.
     * @param n Number of columns.
     */
    public DenseMatrix(int m, int n) {
        this(m, n, new double[m * n], false);
    }

    /**
     * Construct a matrix from a 1-D array. The data in the array should organize in column major.
     *
     * @param m Number of rows.
     * @param n Number of cols.
     * @param data One-dimensional array of doubles.
     */
    public DenseMatrix(int m, int n, double[] data) {
        this(m, n, data, false);
    }

    /**
     * Construct a matrix from a 1-D array. The data in the array is organized in column major or in
     * row major, which is specified by parameter 'inRowMajor'
     *
     * @param m Number of rows.
     * @param n Number of cols.
     * @param data One-dimensional array of doubles.
     * @param inRowMajor Whether the matrix in 'data' is in row major format.
     */
    public DenseMatrix(int m, int n, double[] data, boolean inRowMajor) {
        assert (data.length == m * n);
        this.m = m;
        this.n = n;
        if (inRowMajor) {
            toColumnMajor(m, n, data);
        }
        this.data = data;
    }

    /**
     * Get a single element.
     *
     * @param i Row index.
     * @param j Column index.
     * @return matA(i, j)
     * @throws ArrayIndexOutOfBoundsException
     */
    public double get(int i, int j) {
        return data[j * m + i];
    }

    /**
     * Get the data array of this matrix.
     *
     * @return the data array of this matrix.
     */
    public double[] getData() {
        return this.data;
    }

    /**
     * Set a single element.
     *
     * @param i Row index.
     * @param j Column index.
     * @param s A(i,j).
     * @throws ArrayIndexOutOfBoundsException
     */
    public void set(int i, int j, double s) {
        data[j * m + i] = s;
    }

    /**
     * Get the number of rows.
     *
     * @return the number of rows.
     */
    public int numRows() {
        return m;
    }

    /**
     * Get the number of columns.
     *
     * @return the number of columns.
     */
    public int numCols() {
        return n;
    }

    /**
     * Create a new matrix by transposing current matrix.
     *
     * <p>Use cache-oblivious matrix transpose algorithm.
     */
    public DenseMatrix transpose() {
        DenseMatrix mat = new DenseMatrix(n, m);
        int m0 = m;
        int n0 = n;
        int barrierSize = 16384;
        while (m0 * n0 > barrierSize) {
            if (m0 >= n0) {
                m0 /= 2;
            } else {
                n0 /= 2;
            }
        }
        for (int i0 = 0; i0 < m; i0 += m0) {
            for (int j0 = 0; j0 < n; j0 += n0) {
                for (int i = i0; i < i0 + m0 && i < m; i++) {
                    for (int j = j0; j < j0 + n0 && j < n; j++) {
                        mat.set(j, i, this.get(i, j));
                    }
                }
            }
        }
        return mat;
    }

    /** Converts the data layout in "data" from row major to column major. */
    private static void toColumnMajor(int m, int n, double[] data) {
        if (m == n) {
            for (int i = 0; i < m; i++) {
                for (int j = i + 1; j < m; j++) {
                    int pos0 = j * m + i;
                    int pos1 = i * m + j;
                    double t = data[pos0];
                    data[pos0] = data[pos1];
                    data[pos1] = t;
                }
            }
        } else {
            DenseMatrix temp = new DenseMatrix(n, m, data, false);
            System.arraycopy(temp.transpose().data, 0, data, 0, data.length);
        }
    }
}
