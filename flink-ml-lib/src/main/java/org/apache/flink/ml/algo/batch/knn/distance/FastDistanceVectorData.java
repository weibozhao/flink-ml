package org.apache.flink.ml.algo.batch.knn.distance;

import org.apache.flink.ml.common.linalg.DenseVector;
import org.apache.flink.ml.common.linalg.SparseVector;
import org.apache.flink.ml.common.linalg.Vector;
import org.apache.flink.ml.common.linalg.VectorUtil;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import java.util.HashMap;
import java.util.Map;

/**
 * Save the data for calculating distance fast. The FastDistanceMatrixData
 */
public class FastDistanceVectorData extends BaseFastDistanceData {
	private static final long serialVersionUID = 2149044894420822785L;
	/**
	 * Stores the vector(sparse or dense).
	 */
	final Vector vector;

	/**
	 * Stores some extra info extracted from the vector. For example, if we want to save the L1 norm and L2 norm of the
	 * vector, then the two values are viewed as a two-dimension label vector.
	 */
	DenseVector label;

	/**
	 * Constructor, initialize the vector data.
	 *
	 * @param vec vector.
	 */
	public FastDistanceVectorData(Vector vec) {
		this(vec, null);
	}

	/**
	 * Constructor, initialize the vector data and extra info.
	 *
	 * @param vec vector.
	 * @param row extra info besides the vector.
	 */
	public FastDistanceVectorData(Vector vec, Row row) {
		super(null == row ? null : new Row[] {row});
		Preconditions.checkNotNull(vec, "Vector should not be null!");
		this.vector = vec;
	}

	/**
	 * constructor.
	 * @param vectorData vector data.
	 */
	public FastDistanceVectorData(FastDistanceVectorData vectorData) {
		super(vectorData);
		if (vectorData.vector instanceof SparseVector) {
			this.vector = ((SparseVector) vectorData.vector).clone();
		} else {
			this.vector = ((DenseVector) vectorData.vector).clone();
		}
		this.label = (null == vectorData.label ? null : vectorData.label.clone());
	}

	public Vector getVector() {
		return vector;
	}

	public DenseVector getLabel() {
		return label;
	}

	/**
	 * serialization of FastDistanceVectorData.
	 * @return json string.
	 */
	@Override
	public String toString() {
		Map <String, Object> params = new HashMap <>(3);
		params.put("vector", vector.toString());
		params.put("label", pGson.toJson(label));
		if (rows != null) {
			params.put("rows", pGson.toJson(rows[0]));
		}
		return pGson.toJson(params);
	}

	/**
	 * deserialization of FastDistanceVectorData.
	 *
	 * @param modelStr string of model serialization.
	 * @return FastDistanceVectorData
	 */
	public static FastDistanceVectorData fromString(String modelStr) {
		Map <String, Object> params = pGson.fromJson(modelStr, HashMap.class);
		Row row = parseRowCompatible(params);
		String vector = (String) params.get("vector");
		DenseVector label = pGson.fromJson((String) params.get("label"), DenseVector.class);
		FastDistanceVectorData vectorData =
			new FastDistanceVectorData(VectorUtil.parse(vector), row);
		vectorData.label = label;
		return vectorData;
	}
}
