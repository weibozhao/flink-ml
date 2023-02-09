package org.apache.flink.ml.linalg;

import org.netlib.util.intW;

import java.util.Arrays;

/** A normal equation is A^T * A * x = A^T * b, where A * x = b is a lease square problem. */
public class NormalEquationSolver {
    private static final dev.ludovic.netlib.NativeBLAS NATIVE_BLAS =
            dev.ludovic.netlib.NativeBLAS.getInstance();
    private static final dev.ludovic.netlib.LAPACK LAPACK =
            dev.ludovic.netlib.lapack.F2jLAPACK.getInstance();
    /** Rank of the equation. */
    private final int n;

    /** A^T * A. */
    private final DenseMatrix ata;

    /** A^T * b. */
    private final DenseVector atb;

    NNLS nnls;
    /**
     * The constructor.
     *
     * @param n Rank of the equation.
     */
    public NormalEquationSolver(int n) {
        this.n = n;
        this.ata = new DenseMatrix(n, n);
        this.atb = new DenseVector(n);
        nnls = new NNLS();
    }

    /**
     * Add coefficients to the normal equation.
     *
     * @param a A row of matrix "A".
     * @param b An element of right hand side "b".
     * @param c The scale factor of "a".
     */
    public void add(DenseVector a, double b, double c) {
        // ata += c * a.t * a
        NATIVE_BLAS.dger(n, n, c, a.values, 1, a.values, 1, this.ata.values, n);

        // atb += b * a.t
        BLAS.axpy(b, a, this.atb);
    }

    /** Reset the system to zeros. */
    public void reset() {
        Arrays.fill(ata.values, 0.);
        Arrays.fill(atb.values, 0.);
    }

    /** Merge with another A^T*A. */
    public void merge(DenseMatrix otherAta) {
        NATIVE_BLAS.daxpy(ata.values.length, 1.0, otherAta.values, 1, ata.values, 1);
    }

    /** Merge with another NormalEquation. */
    public void merge(NormalEquationSolver otherEq) {
        merge(otherEq.ata);
        BLAS.axpy(1.0, otherEq.atb, this.atb);
    }

    /** Regularize the system by adding "lambda" to diagonals. */
    public void regularize(double lambda) {
        for (int i = 0; i < n; i++) {
            this.ata.add(i, i, lambda);
        }
    }

    /**
     * Solve the system. After solving the system, the result is returned in <code>x</code>, and the
     * data in <code>ata</code> and <code>atb</code> will be reset to zeros.
     *
     * @param x For holding the result.
     * @param nonNegative Whether to enforce non-negative constraint.
     */
    public void solve(DenseVector x, boolean nonNegative) {
        if (nonNegative) {
            nnls.initialize(n);
            double[] ret = nnls.solve(ata.values, atb.values);
            System.arraycopy(ret, 0, x.values, 0, n);
        } else {
            int n = ata.numCols();
            int nrhs = atb.size();
            intW info = new intW(0);
            // require(A.isSymmetric, "A is not symmetric")
            LAPACK.dposv("U", n, nrhs, ata.values, n, atb.values, n, info);
            // check solution
            if (info.val > 0) {
                throw new RuntimeException("A is not positive definite.");
            } else if (info.val < 0) {
                throw new RuntimeException("Invalid input to lapack routine.");
            }
            System.arraycopy(atb.values, 0, x.values, 0, n);
        }
        this.reset();
    }
}
