package com.example.myapplication;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class CTC {
    private Map<String, Integer> dict;
    private List<String> character;
    private Map<String, List<String>> separatorList;
    private List<Integer> ignoreIdx;
    private Map<String, List<String>> dictList;

    public CTC(String character, Map<String, List<String>> separatorList, Map<String, String> dictPathList) {
        List<String> dictCharacter = new ArrayList<>();
        for (char c : character.toCharArray()) {
            dictCharacter.add(String.valueOf(c));
        }

        this.dict = new HashMap<>();
        for (int i = 0; i < dictCharacter.size(); i++) {
            this.dict.put(dictCharacter.get(i), i + 1);
        }

        this.character = new ArrayList<>();
        this.character.add("[blank]");
        this.character.addAll(dictCharacter);

        this.separatorList = separatorList;
        List<String> separatorChar = new ArrayList<>();
        for (Map.Entry<String, List<String>> entry : separatorList.entrySet()) {
            separatorChar.addAll(entry.getValue());
        }
        this.ignoreIdx = new ArrayList<>();
        this.ignoreIdx.add(0);
        for (int i = 0; i < separatorChar.size(); i++) {
            this.ignoreIdx.add(i + 1);
        }

        if (separatorList.isEmpty()) {
            List<String> dictList = new ArrayList<>();
            for (Map.Entry<String, String> entry : dictPathList.entrySet()) {
                try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(entry.getValue()), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        dictList.add(line);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            this.dictList = new HashMap<>();
            this.dictList.put("default", dictList);
        } else {
            this.dictList = new HashMap<>();
            for (Map.Entry<String, String> entry : dictPathList.entrySet()) {
                List<String> wordCount = new ArrayList<>();
                try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(entry.getValue()), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        wordCount.add(line);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
                this.dictList.put(entry.getKey(), wordCount);
            }
        }
    }

    public Pair<int[], int[]> encode(List<String> text, int batchMaxLength) {
        int[] length = new int[text.size()];
        for (int i = 0; i < text.size(); i++) {
            length[i] = text.get(i).length();
        }
        StringBuilder concatenatedText = new StringBuilder();
        for (String s : text) {
            concatenatedText.append(s);
        }
        int[] textIndices = new int[concatenatedText.length()];
        for (int i = 0; i < concatenatedText.length(); i++) {
            textIndices[i] = this.dict.get(String.valueOf(concatenatedText.charAt(i)));
        }
        return new Pair<>(textIndices, length);
    }

    public List<String> decodeGreedy(int[] textIndex, int[] length) {
        List<String> texts = new ArrayList<>();
        int index = 0;
        for (int l : length) {
            int[] t = new int[l];
            System.arraycopy(textIndex, index, t, 0, l);
            boolean[] a = new boolean[l];
            a[0] = true;
            for (int i = 1; i < l; i++) {
                a[i] = t[i] != t[i - 1];
            }
            boolean[] b = new boolean[l];
            for (int i = 0; i < l; i++) {
                b[i] = !this.ignoreIdx.contains(t[i]);
            }
            boolean[] c = new boolean[l];
            for (int i = 0; i < l; i++) {
                c[i] = a[i] && b[i];
            }
            StringBuilder text = new StringBuilder();
            for (int i = 0; i < l; i++) {
                if (c[i]) {
                    text.append(this.character.get(t[i]));
                }
            }
            texts.add(text.toString());
            index += l;
        }
        return texts;
    }

    private int argmax(float[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static List<Integer> consecutive(int[] data, String mode, int stepsize) {
        List<List<Integer>> group = new ArrayList<>();
        List<Integer> currentGroup = new ArrayList<>();
        currentGroup.add(data[0]);

        for (int i = 1; i < data.length; i++) {
            if (data[i] - data[i - 1] == stepsize) {
                currentGroup.add(data[i]);
            } else {
                group.add(new ArrayList<>(currentGroup));
                currentGroup.clear();
                currentGroup.add(data[i]);
            }
        }
        group.add(currentGroup);

        group = group.stream().filter(item -> item.size() > 0).collect(Collectors.toList());

        List<Integer> result = new ArrayList<>();
        if (mode.equals("first")) {
            for (List<Integer> l : group) {
                result.add(l.get(0));
            }
        } else if (mode.equals("last")) {
            for (List<Integer> l : group) {
                result.add(l.get(l.size() - 1));
            }
        }
        return result;
    }

    public static List<WordSegment> wordSegmentation(int[] mat) {
        Map<String, List<Integer>> separatorIdx = new HashMap<>();
        separatorIdx.put("en", Arrays.asList(3, 4));  // Only English separators
        List<Integer> separatorIdxList = new ArrayList<>();
        separatorIdxList.add(3);
        separatorIdxList.add(4);
        List<WordSegment> result = new ArrayList<>();
        List<int[]> sepList = new ArrayList<>();
        int startIdx = 0;
        String sepLang = "";

        for (int sepIdx : separatorIdxList) {
            String mode = (sepIdx % 2 == 0) ? "first" : "last";
            int[] indices = IntStream.range(0, mat.length).filter(i -> mat[i] == sepIdx).toArray();
            List<Integer> a = consecutive(indices, mode, 1);
            for (int item : a) {
                sepList.add(new int[]{item, sepIdx});
            }
        }

        sepList.sort(Comparator.comparingInt(a -> a[0]));

        for (int[] sep : sepList) {
            for (String lang : separatorIdx.keySet()) {
                if (sep[1] == separatorIdx.get(lang).get(0)) {  // start lang
                    sepLang = lang;
                    int sepStartIdx = sep[0];
                } else if (sep[1] == separatorIdx.get(lang).get(1)) {  // end lang
                    if (sepLang.equals(lang)) {  // check if last entry if the same start lang
                        int sepStartIdx = sep[0];
                        WordSegment newSegPair = new WordSegment(lang, sepStartIdx + 1, sep[0] - 1);
                        if (sepStartIdx > startIdx) {
                            result.add(new WordSegment("", startIdx, sepStartIdx - 1));
                        }
                        startIdx = sep[0] + 1;
                        result.add(newSegPair);
                    }
                    sepLang = "";  // reset
                }
            }
        }

        if (startIdx <= mat.length - 1) {
            result.add(new WordSegment("", startIdx, mat.length - 1));
        }
        return result;
    }

    private static class WordSegment {
        String lang;
        int start;
        int end;

        WordSegment(String lang, int start, int end) {
            this.lang = lang;
            this.start = start;
            this.end = end;
        }
    }

    private static class Pair<K, V> {
        private K key;
        private V value;

        Pair(K key, V value) {
            this.key = key;
            this.value = value;
        }

        public K getKey() {
            return key;
        }

        public V getValue() {
            return value;
        }
    }
}