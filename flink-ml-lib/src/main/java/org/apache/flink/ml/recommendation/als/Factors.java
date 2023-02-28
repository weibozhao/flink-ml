package org.apache.flink.ml.recommendation.als;

import java.io.Serializable;

/**
 * Factors of a user or an item.
 */
public class Factors {
	public byte identity; // 0->user, 1->item
	public long nodeId; // userId or itemId
	public float[] factors;

	public Factors() {
	}

	/**
	 * Since we use double precision to solve the least square problem, we need to convert the
	 * factors to double array.
	 */
	public void getFactorsAsDoubleArray(double[] buffer) {
		for (int i = 0; i < factors.length; i++) {
			buffer[i] = factors[i];
		}
	}

	public void copyFactorsFromDoubleArray(double[] buffer) {
		if (factors == null) {
			factors = new float[buffer.length];
		}
		for (int i = 0; i < buffer.length; i++) {
			factors[i] = (float) buffer[i];
		}
	}
}
