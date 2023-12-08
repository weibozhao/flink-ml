package org.apache.flink.ml.common.fm;

import org.apache.flink.streaming.api.datastream.DataStream;

public class DataSet<T>  {
	public DataSet(DataStream <T> data) {
		this.data = data;
	}
	public DataStream<T> data;
}
