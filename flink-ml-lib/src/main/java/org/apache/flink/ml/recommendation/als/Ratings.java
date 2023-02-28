package org.apache.flink.ml.recommendation.als;

/** All ratings of a user or an item. */
public class Ratings {
	public Ratings() {
	}

	public byte identity; // 0->user, 1->item
	public long nodeId; // userId or itemId
	public long[] neighbors;
	public float[] ratings;
}
