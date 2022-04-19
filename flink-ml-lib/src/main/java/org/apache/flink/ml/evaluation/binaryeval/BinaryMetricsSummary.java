package org.apache.flink.ml.evaluation.binaryeval;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.util.Preconditions;

import java.util.ArrayList;
import java.util.List;

/** Save the evaluation data for binary classification. */
public class BinaryMetricsSummary {
    private static final long serialVersionUID = 4614108912380382179L;

    /** Decision threshold */
    double decisionThreshold;

    /** The count of samples. */
    long total;

    /** Area under Roc */
    double auc;

    /** Area under Lorenz */
    double gini;

    /** Area under PRC */
    double prc;

    /** KS */
    double ks;

    double accuracy;

    double kappa;

    double precision;

    double f1;

    double fnr, fpr, tnr, tpr;
    /** Logloss = sum_i{sum_j{y_ij * log(p_ij)}} */
    double logLoss;

   ConfusionMatrix confusionMatrix;

    public BinaryMetricsSummary(double logLoss, long total, double auc) {
        this(0.5, logLoss, total, auc);
    }

    public BinaryMetricsSummary(double decisionThreshold, double logLoss, long total, double auc) {
        this.decisionThreshold = decisionThreshold;
        this.logLoss = logLoss;
        this.total = total;
        this.auc = auc;
    }

    public BinaryMetricsSummary merge(BinaryMetricsSummary binaryClassMetrics) {
        if (null == binaryClassMetrics) {
            return this;
        }
        Preconditions.checkState(
                Double.compare(auc, binaryClassMetrics.auc) == 0, "Auc not equal!");
        logLoss += binaryClassMetrics.logLoss;
        total += binaryClassMetrics.total;
        ks = Math.max(ks, binaryClassMetrics.ks);
        prc += binaryClassMetrics.prc;
        gini += binaryClassMetrics.gini;
        if (binaryClassMetrics.confusionMatrix != null) {
            this.confusionMatrix = binaryClassMetrics.confusionMatrix;
        }
        return this;
    }

    /**
     * TPR = TP / (TP + FN)
     */
    public static double truePositiveRate(ConfusionMatrix matrix) {
        double denominator =
            matrix.numTruePositive(1) + matrix.numFalseNegative(1);
        return denominator == 0 ? 0.0 : matrix.numTruePositive(1) / denominator;
    }

    /**
     * TNR = TN / (FP + TN)
     */
    public static double trueNegativeRate(ConfusionMatrix matrix) {
        double denominator =
            matrix.numFalsePositive(1) + matrix.numTrueNegative(1);
        return denominator == 0 ? 0.0 : matrix.numTrueNegative(1) / denominator;
    }

    /**
     * FPR = FP / (FP + TN)
     */
    public static double falsePositiveRate(ConfusionMatrix matrix) {
        double denominator =
            matrix.numFalsePositive(1) + matrix.numTrueNegative(1);
        return denominator == 0 ? 0.0 : matrix.numFalsePositive(1) / denominator;
    }

    /**
     * FNR = FN / (TP + FN)
     */
    public static double falseNegativeRate(ConfusionMatrix matrix) {
        double denominator =
            matrix.numTruePositive(1) + matrix.numFalseNegative(1);
        return denominator == 0 ? 0.0 : matrix.numFalseNegative(1) / denominator;
    }

    /**
     * p_a = (TP+TN)/total
     *
     * <p>p_e = ((TN+FP)(TN+FN)+(FN+TP)(FP+TP))/total/total
     *
     * <p>kappa = (p_a - p_e)/(1 - p_e)
     */
    public static double kappa(ConfusionMatrix matrix) {
        double total =
            matrix.numFalseNegative(1)
                + matrix.numFalsePositive(1)
                + matrix.numTrueNegative(1)
                + matrix.numTruePositive(1);

        double pa = matrix.numTruePositive(1) + matrix.numTrueNegative(1);
        pa /= total;

        double pe =
            (matrix.numTruePositive(1) + matrix.numFalseNegative(1))
                * (matrix.numTruePositive(1)
                + matrix.numFalsePositive(1));
        pe +=
            (matrix.numTrueNegative(1) + matrix.numFalsePositive(1))
                * (matrix.numTrueNegative(1)
                + matrix.numFalseNegative(1));
        pe /= (total * total);
        if (pe < 1) {
            return (pa - pe) / (1 - pe);
        } else {
            return 1.0;
        }
    }

    /**
     * PRECISION: TP / (TP + FP)
     */
    public static Double precision(ConfusionMatrix matrix) {
        double denominator =
            matrix.numTruePositive(1) + matrix.numFalsePositive(1);
        return denominator == 0 ? 1.0 : matrix.numTruePositive(1) / denominator;
    }

    /**
     * F1: 2 * Precision * Recall / (Precision + Recall)
     *
     * <p>F1: 2 * TP / (2TP + FP + FN)
     */
    public static Double f1(ConfusionMatrix matrix) {
        double denominator =
            2 * matrix.numTruePositive(1)
                + matrix.numFalsePositive(1)
                + matrix.numFalseNegative(1);
        return denominator == 0 ? 0.0 : 2 * matrix.numTruePositive(1) / denominator;
    }

    /**
     * ACCURACY: (TP + TN) / Total
     */
    public static Double accuracy(ConfusionMatrix matrix) {
        double total =
            matrix.numFalseNegative(1)
                + matrix.numFalsePositive(1)
                + matrix.numTrueNegative(1)
                + matrix.numTruePositive(1);

        return (matrix.numTruePositive(1) + matrix.numTrueNegative(1))
            / total;
    }
}
