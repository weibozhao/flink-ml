package org.apache.flink.ml.classification.ftrl;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.util.Preconditions;

import scala.collection.immutable.VectorIterator;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class Ftrl implements Estimator <Ftrl, FtrlModel>, FtrlParams <Ftrl> {
	private final Map <Param <?>, Object> paramMap = new HashMap <>();

	public Ftrl() {
		ParamUtils.initializeMapWithDefaultValues(paramMap, this);
	}

	@Override
	public FtrlModel fit(Table... inputs) {
		Preconditions.checkArgument(inputs.length == 1);

		StreamTableEnvironment tEnv =
			(StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
		DataStream <Tuple2 <DenseVector, Double>> trainData =
			tEnv.toDataStream(inputs[0])
				.map(row -> Tuple2
					.of((DenseVector) row.getField(getFeaturesCol()), (double) row.getField(getLabelCol())));

		FtrlModel model = new FtrlModel().setModelData(null);
		ReadWriteUtils.updateExistingParams(model, paramMap);
		return model;
	}

	@Override
	public void save(String path) throws IOException {

	}

	@Override
	public Map <Param <?>, Object> getParamMap() {
		return null;
	}

	private void updateProc(DenseVector weights, Vector featureVector, double label, int index) {
		double p = 0.0;
		VectorIterator vi = featureVector.iterator();
		while (vi.hasNext()) {
			int i = vi.getIndex();
			if (Math.abs(Z[index][i]) <= L1) {
				weights.set(i, 0.0);
			} else {
				weights.set(i, ((Z[index][i] < 0 ? -1 : 1) * L1 - Z[index][i]) / ((beta + Math.sqrt(N[index][i]))
					/ alpha + L2));
			}
			p += weights.get(i) * vi.getValue();
			vi.next();
		}
		// eta
		p = 1 / (1 + Math.exp(-p));

		// update
		vi = featureVector.iterator();
		while (vi.hasNext()) {
			int i = vi.getIndex();
			double g = (p - label) * vi.getValue();
			double sigma = (Math.sqrt(N[index][i] + g * g) - Math.sqrt(N[index][i])) / alpha;
			Z[index][i] += g - sigma * weights.get(i);
			N[index][i] += g * g;
			vi.next();
		}
	}
}
