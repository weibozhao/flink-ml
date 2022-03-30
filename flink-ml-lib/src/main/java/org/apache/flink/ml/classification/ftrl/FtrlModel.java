package org.apache.flink.ml.classification.ftrl;

import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.param.Param;
import org.apache.flink.table.api.Table;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class FtrlModel implements Model <FtrlModel>, FtrlParams <FtrlModel> {
	private final Map <Param <?>, Object> paramMap = new HashMap <>();
	private Table modelDataTable;

	@Override
	public Table[] transform(Table... inputs) {
		return new Table[0];
	}

	@Override
	public void save(String path) throws IOException {

	}

	@Override
	public Map <Param <?>, Object> getParamMap() {
		return null;
	}

	@Override
	public FtrlModel setModelData(Table... inputs) {
		modelDataTable = inputs[0];
		return this;
	}

	@Override
	public Table[] getModelData() {
		return new Table[] {modelDataTable};
	}
}
