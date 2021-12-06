package org.apache.flink.ml.classification.knn;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.types.DataType;
import org.apache.flink.types.Row;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import java.io.IOException;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** knn model data, which stores the data used to calculate the distances between nodes. */
public class KnnModelData implements Serializable, Cloneable {
    private final List<FastDistanceMatrixData> dictData;
    private final FastDistance fastDistance;
    private final Comparator<? super Tuple2<Double, Object>> comparator;

    /**
     * Constructor.
     *
     * @param list FastDistanceMatrixData list.
     * @param fastDistance used to accelerate the speed of calculating distance.
     */
    public KnnModelData(List<FastDistanceMatrixData> list, FastDistance fastDistance) {
        this.dictData = list;
        this.fastDistance = fastDistance;
        comparator = Comparator.comparingDouble(o -> -o.f0);
    }

    /**
     * Get comparator.
     *
     * @return Comparator.
     */
    public Comparator<? super Tuple2<Double, Object>> getQueueComparator() {
        return comparator;
    }

    /**
     * Get dictionary data size.
     *
     * @return Dictionary data size.
     */
    public Integer getLength() {
        return dictData.size();
    }

    /**
     * Get fastDistance.
     *
     * @return fastDistance
     */
    public FastDistance getFastDistance() {
        return fastDistance;
    }

    public List<FastDistanceMatrixData> getDictData() {
        return dictData;
    }

    /** Encoder for the Knn model data. */
    public static class ModelDataEncoder implements Encoder<Row> {
        @Override
        public void encode(Row modelData, OutputStream outputStream) {
            Kryo kryo = new Kryo();
            Output output = new Output(outputStream);
            List<Object> objs = new ArrayList<>();
            for (int i = 0; i < modelData.getArity(); ++i) {
                objs.add(modelData.getField(i));
            }
            kryo.writeObject(output, objs);
            output.flush();
        }
    }

    /** Decoder for the Knn model data. */
    public static class ModelDataStreamFormat extends SimpleStreamFormat<Row> {
        private final DataType idType;

        public ModelDataStreamFormat(DataType idType) {
            this.idType = idType;
        }

        @Override
        public Reader<Row> createReader(Configuration config, FSDataInputStream stream) {
            return new Reader<Row>() {
                private final Kryo kryo = new Kryo();
                private final Input input = new Input(stream);

                @Override
                public Row read() {
                    if (input.eof()) {
                        return null;
                    }
                    List<Object> objs = kryo.readObject(input, ArrayList.class);
                    Row ret = new Row(objs.size());
                    for (int i = 0; i < objs.size(); ++i) {
                        ret.setField(i, objs.get(i));
                    }
                    return ret;
                }

                @Override
                public void close() throws IOException {
                    stream.close();
                }
            };
        }

        @Override
        public TypeInformation<Row> getProducedType() {
            return new RowTypeInfo(
                    new TypeInformation[] {
                        Types.STRING,
                        TypeInformation.of(idType.getLogicalType().getDefaultConversion())
                    },
                    new String[] {"DATA", "KNN_LABEL_TYPE"});
        }
    }

    public static Schema getModelSchema(DataType idType) {
        return Schema.newBuilder()
                .column("DATA", DataTypes.STRING())
                .column("KNN_LABEL_TYPE", idType)
                .build();
    }
}
