package org.apache.flink.ml.algo.batch.knn;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.linalg.DenseMatrix;
import org.apache.flink.ml.common.linalg.VectorUtil;
import org.apache.flink.ml.algo.batch.knn.distance.BaseFastDistance;
import org.apache.flink.ml.algo.batch.knn.distance.BaseFastDistanceData;
import org.apache.flink.ml.algo.batch.knn.distance.FastDistanceVectorData;
import org.apache.flink.shaded.curator4.com.google.common.collect.ImmutableMap;
import org.apache.flink.table.types.DataType;
import org.apache.flink.types.Row;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.TreeMap;

import static org.apache.flink.ml.algo.batch.knn.distance.BaseFastDistanceData.pGson;

/** knn model data, which will be used to calculate the distances between nodes. */
public class KnnModelData implements Serializable, Cloneable {
    private static final long serialVersionUID = -2940551481683238630L;
    private final List<BaseFastDistanceData> dictData;
    private final BaseFastDistance fastDistance;
    protected Comparator<? super Tuple2<Double, Object>> comparator;
    private DataType idType;

    /**
     * constructor.
     *
     * @param list BaseFastDistanceData list.
     * @param fastDistance used to accelerate the speed of calculating distance.
     */
    public KnnModelData(List<BaseFastDistanceData> list, BaseFastDistance fastDistance) {
        this.dictData = list;
        this.fastDistance = fastDistance;
        comparator = Comparator.comparingDouble(o -> -o.f0);
    }
    /**
     * set id type.
     *
     * @param idType id type.
     */
    public void setIdType(DataType idType) {
        this.idType = idType;
    }

    /**
     * find the nearest topN neighbors from whole nodes.
     *
     * @param input input node.
     * @param topN top N.
     * @param radius the parameter to describe the range to find neighbors.
     * @return
     */
    public String findNeighbor(Object input, Integer topN, Double radius) {
        PriorityQueue<Tuple2<Double, Object>> priorityQueue =
                new PriorityQueue<>(this.getQueueComparator());
        search(input, topN, Tuple2.of(radius, null), priorityQueue);

        List<Object> items = new ArrayList<>();
        List<Double> metrics = new ArrayList<>();
        while (!priorityQueue.isEmpty()) {
            Tuple2<Double, Object> result = priorityQueue.poll();
            items.add(KnnUtils.castTo(result.f1, idType));
            metrics.add(result.f0);
        }
        Collections.reverse(items);
        Collections.reverse(metrics);
        priorityQueue.clear();
        return serializeResult(items, ImmutableMap.of("METRIC", metrics));
    }

    /**
     * @param input input node.
     * @param topN top N.
     * @param radius the parameter to describe the range to find neighbors.
     * @param priorityQueue priority queue.
     */
    protected void search(
            Object input,
            Integer topN,
            Tuple2<Double, Object> radius,
            PriorityQueue<Tuple2<Double, Object>> priorityQueue) {
        Object sample = prepareSample(input);
        Tuple2<Double, Object> head = null;
        for (int i = 0; i < getLength(); i++) {
            ArrayList<Tuple2<Double, Object>> values = computeDistance(sample, i, radius);
            if (null == values || values.size() == 0) {
                continue;
            }
            for (Tuple2<Double, Object> currentValue : values) {
                if (null == topN) {
                    priorityQueue.add(Tuple2.of(currentValue.f0, currentValue.f1));
                } else {
                    head = KnnUtils.updateQueue(priorityQueue, topN, currentValue, head);
                }
            }
        }
    }

    /**
     * get comparator.
     *
     * @return comparator.
     */
    private Comparator<? super Tuple2<Double, Object>> getQueueComparator() {
        return comparator;
    }

    /**
     * get dictionary data size.
     *
     * @return dictionary data size.
     */
    private Integer getLength() {
        return dictData.size();
    }

    /**
     * prepare sample.
     *
     * @param input sample to parse.
     * @return
     */
    private Object prepareSample(Object input) {
        return fastDistance.prepareVectorData(Tuple2.of(VectorUtil.parse(input.toString()), null));
    }

    private ArrayList<Tuple2<Double, Object>> computeDistance(
            Object input, Integer index, Tuple2<Double, Object> radius) {
        BaseFastDistanceData data = dictData.get(index);
        DenseMatrix res = fastDistance.calc((FastDistanceVectorData) input, data);
        ArrayList<Tuple2<Double, Object>> list = new ArrayList<>(0);
        Row[] curRows = data.getRows();
        for (int i = 0; i < data.getRows().length; i++) {
            Tuple2<Double, Object> tuple = Tuple2.of(res.getData()[i], curRows[i].getField(0));
            if (null == radius
                    || radius.f0 == null
                    || this.getQueueComparator().compare(radius, tuple) <= 0) {
                list.add(tuple);
            }
        }
        return list;
    }

    /**
     * serialize result to json format.
     *
     * @param objectValue the nearest nodes found.
     * @param others the metric of nodes.
     * @return serialize result.
     */
    private static String serializeResult(
            List<Object> objectValue, Map<String, List<Double>> others) {
        final String id = "ID";
        Map<String, String> result =
                new TreeMap<>(
                        (o1, o2) -> {
                            if (id.equals(o1) && id.equals(o2)) {
                                return 0;
                            } else if (id.equals(o1)) {
                                return -1;
                            } else if (id.equals(o2)) {
                                return 1;
                            }

                            return o1.compareTo(o2);
                        });

        result.put(id, pGson.toJson(objectValue));

        if (others != null) {
            for (Map.Entry<String, List<Double>> other : others.entrySet()) {
                result.put(other.getKey(), pGson.toJson(other.getValue()));
            }
        }
        return pGson.toJson(result);
    }
}
