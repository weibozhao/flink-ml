package org.apache.flink.ml.common.fm;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Model data of {@link FmModelServable}.
 *
 * <p>This class also provides methods to encoding or decoding the model data.
 */
public class FmModelData {

    public List<Tuple2<Long, float[]>> factors;
    public final int[] dim;
    public boolean isReg;

    public FmModelData(List<Tuple2<Long, float[]>> factors, int[] dim, boolean isReg) {
        this.factors = factors;
        this.dim = dim;
        this.isReg = isReg;
    }

    public void encode(OutputStream outputStream) throws IOException {
        DataOutputView dataOutputView = new DataOutputViewStreamWrapper(outputStream);
        dataOutputView.writeBoolean(isReg);
        for (int e : dim) {
            dataOutputView.writeInt(e);
        }
        dataOutputView.writeInt(this.factors.size());
        if (this.factors.size() > 0) {
            dataOutputView.writeInt(this.factors.get(0).f1.length);
            for (Tuple2<Long, float[]> factor : this.factors) {
                dataOutputView.writeLong(factor.f0);
                float[] values = factor.f1;
                for (float value : values) {
                    dataOutputView.writeFloat(value);
                }
            }
        }
    }

    public static FmModelData decode(InputStream inputStream) throws IOException {
        FmModelData fmModelData = null;

        DataInputViewStreamWrapper inputViewStreamWrapper =
                new DataInputViewStreamWrapper(inputStream);
        while (true) {
            try {
                boolean isReg = inputViewStreamWrapper.readBoolean();
                int[] dim = new int[3];
                dim[0] = inputViewStreamWrapper.readInt();
                dim[1] = inputViewStreamWrapper.readInt();
                dim[2] = inputViewStreamWrapper.readInt();
                int sizeFactor = inputViewStreamWrapper.readInt();
                int rank = inputViewStreamWrapper.readInt();
                List<Tuple2<Long, float[]>> factorList = new ArrayList<>(sizeFactor);
                for (int i = 0; i < sizeFactor; ++i) {
                    long id = inputViewStreamWrapper.readLong();
                    float[] factors = new float[rank];
                    for (int j = 0; j < rank; ++j) {
                        factors[j] = inputViewStreamWrapper.readFloat();
                    }
                    factorList.add(Tuple2.of(id, factors));
                }
                if (fmModelData == null) {
                    fmModelData = new FmModelData(factorList, dim, isReg);
                } else {
                    fmModelData.factors.addAll(factorList);
                }
            } catch (EOFException e) {
                return fmModelData;
            }
        }
    }
}
