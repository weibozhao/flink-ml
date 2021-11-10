package org.apache.flink.ml.params;

import org.apache.flink.ml.param.Param;

import org.apache.commons.lang3.StringUtils;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;

/** Util for the parameters. */
public class ParamUtil {

    private static Stream<Field> getAllFields(Class cls) {
        final List<Class> ret = new ArrayList<>();
        ret.add(cls);
        final Set<Class> hash = new HashSet<>(Collections.singletonList(cls));
        for (int i = 0; i < ret.size(); ++i) {
            Class cur = ret.get(i);
            Optional.ofNullable(cur.getSuperclass()).map(ret::add);
            Stream.of(cur.getInterfaces())
                    .filter(x -> !hash.contains(x))
                    .forEach(
                            x -> {
                                ret.add(x);
                                hash.add(x);
                            });
        }
        return ret.stream().flatMap(x -> Stream.of(x.getDeclaredFields()));
    }

    private static void printOneRow(String[] cells, int[] maxLength) {
        System.out.print("|");
        for (int i = 0; i < cells.length; ++i) {
            System.out.print(" ");
            System.out.print(cells[i]);
            System.out.print(StringUtils.repeat(" ", maxLength[i] - cells[i].length()));
            System.out.print(" |");
        }
        System.out.println();
    }

    /**
     * Convert string to enum, and throw exception.
     *
     * @param enumeration enum class
     * @param search search item
     * @param paramName param name
     * @param <T> class
     * @return enum
     */
    public static <T extends Enum<?>> T searchEnum(
            Class<T> enumeration, String search, String paramName) {
        return searchEnum(enumeration, search, paramName, null);
    }

    /**
     * Convert string to enum, and throw exception.
     *
     * @param paramInfo paramInfo
     * @param search search item
     * @param <T> class
     * @return enum
     */
    public static <T extends Enum<?>> T searchEnum(Param<T> paramInfo, String search) {
        return searchEnum(paramInfo.clazz, search, paramInfo.name);
    }

    /**
     * Convert string to enum, and throw exception.
     *
     * @param enumeration enum class
     * @param search search item
     * @param paramName param name
     * @param opName op name
     * @param <T> class
     * @return enum
     */
    public static <T extends Enum<?>> T searchEnum(
            Class<T> enumeration, String search, String paramName, String opName) {
        if (search == null) {
            return null;
        }
        T[] values = enumeration.getEnumConstants();
        for (T each : values) {
            if (each.name().compareToIgnoreCase(search) == 0) {
                return each;
            }
        }

        StringBuilder sbd = new StringBuilder();
        sbd.append(search).append(" is not member of ").append(paramName);
        if (opName != null && opName.isEmpty()) {
            sbd.append(" of ").append(opName);
        }
        sbd.append(".").append("It maybe ").append(values[0].name());
        for (int i = 1; i < values.length; i++) {
            sbd.append(",").append(values[i].name());
        }
        sbd.append(".");
        throw new RuntimeException(sbd.toString());
    }
}
