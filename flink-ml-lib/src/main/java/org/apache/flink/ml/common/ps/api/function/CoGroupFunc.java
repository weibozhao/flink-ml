package org.apache.flink.ml.common.ps.api.function;

import org.apache.flink.api.common.functions.CoGroupFunction;
import org.apache.flink.api.java.functions.KeySelector;

/** Comments. */
public abstract class CoGroupFunc<IN1, IN2, OUT, K> implements CoGroupFunction<IN1, IN2, OUT> {

    public transient KeySelector<IN1, K> keySelector1;
    public transient KeySelector<IN2, K> keySelector2;

    public CoGroupFunc() {
        keySelector1 = in1 -> computeLeftKey(in1);
        keySelector2 = in2 -> computeRightKey(in2);
    }

    public abstract K computeLeftKey(IN1 t);

    public abstract K computeRightKey(IN2 t);

    // @Override
    // @SuppressWarnings("unchecked")
    // public MLData <?> transform(CommonStage stage, MLData <?> mlData) {
    //	CoGroupStage <?, ?, ?, ?> coGroupStage = (CoGroupStage <?, ?, ?, ?>) stage;
    //	DataStream dataStream = mlData.toDataStream();
    //	dataStream =
    //		dataStream
    //			.coGroup(coGroupStage.coGroupData.toDataStream())
    //			.where(coGroupStage.keySelector1)
    //			.equalTo(coGroupStage.keySelector2)
    //			.window(EndOfStreamWindows.get())
    //			.apply(coGroupStage);
    //	return MLData.fromDataStream(dataStream);
    // }
}
