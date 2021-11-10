package org.apache.flink.ml.algo.batch.knn.distance;

import org.apache.flink.types.Row;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.Serializable;
import java.util.Map;

/**
 * Base class to save the data for calculating distance fast. It has two derived classes:
 * FastDistanceVectorData and FastDistanceMatrixData. FastDistanceVectorData saves only one
 * vector(dense or spase) and FastDistanceMatrixData merges several dense vectors in a matrix.
 */
public abstract class BaseFastDistanceData implements Serializable, Cloneable {
    private static final long serialVersionUID = -6327346472723810463L;
    /**
     * Save the extra info besides the vector. Each vector is related to one row. Thus, for
     * FastDistanceVectorData, the length of <code>rows</code> is one. And for
     * FastDistanceMatrixData, the length of <code>rows</code> is equal to the number of cols of
     * <code>matrix</code>. Besides, the order of the rows are the same with the vectors.
     */
    public final Row[] rows;

    public static Gson pGson =
        new GsonBuilder()
            .serializeNulls()
            .disableHtmlEscaping()
            .serializeSpecialFloatingPointValues()
            .create();

    public Row[] getRows() {
        return rows;
    }

    BaseFastDistanceData(Row[] rows) {
        this.rows = rows;
    }

    BaseFastDistanceData(BaseFastDistanceData fastDistanceData) {
        this.rows = null == fastDistanceData.rows ? null : fastDistanceData.rows.clone();
    }

    public static Row parseRowCompatible(Map <String, Object> params) {
        if (params == null) {
            return null;
        }

        Row row = (Row) params.getOrDefault("rows", null);

        if (row == null) {
            return null;
        }

        if (row.getKind() == null) {
            LegacyRow legacyRow = (LegacyRow) params.getOrDefault("rows", null);

            Object[] objects = legacyRow.getFields();

            row = Row.of(objects);
        }

        return row;
    }

    public static Row[] parseRowArrayCompatible(Map <String, Object> params) {
        if (params == null) {
            return null;
        }

        Row[] rows = pGson.fromJson((String)params.get("rows"), Row[].class);

        if (rows == null) {
            return null;
        }

        if (rows.length == 0) {
            return rows;
        }

        if (rows[0].getKind() == null) {
            LegacyRow[] legacyRow = (LegacyRow[]) params.get("rows");

            rows = new Row[legacyRow.length];

            for (int i = 0; i < legacyRow.length; ++i) {
                rows[i] = Row.of(legacyRow[i].getFields());
            }
        }

        return rows;
    }

    private static final class LegacyRow {
        private final Object[] fields;

        public LegacyRow(Object[] fields) {
            this.fields = fields;
        }

        public Object[] getFields() {
            return fields;
        }
    }
}
