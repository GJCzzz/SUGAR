package com.example.myapplication;

import com.example.myapplication.CTC;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Recognizer {

    static {
        // Load OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static Mat preprocessImage(Mat image, int modelHeight, boolean keepRatioWithPad, double adjustContrast) {
        // Resize and preprocess image
        int imgW = 100; // Example width for resizing
        int imgH = modelHeight; // Example height for resizing

        Mat resizedImage = new Mat();
        Imgproc.resize(image, resizedImage, new Size(imgW, imgH), 0, 0, Imgproc.INTER_LANCZOS4);

        if (adjustContrast > 0) {
            resizedImage = adjustContrastGrey(resizedImage, adjustContrast);
        }

        if (keepRatioWithPad) {
            NormalizePAD transform = new NormalizePAD(new Size(imgW, imgH));
            resizedImage = transform.apply(resizedImage);
        }

        return resizedImage;
    }

    private static Mat adjustContrastGrey(Mat img, double target) {
        Mat newImg = new Mat();
        img.convertTo(newImg, -1, target, 0);
        return newImg;
    }

    public static List<String[]> getText(
            OrtSession session,
            CTC converter,
            List<String> imagePaths,
            int batchMaxLength,
            int ignoreIdx,
            String decoder,
            int beamWidth,
            String device,
            int modelHeight,
            boolean keepRatioWithPad,
            double adjustContrast
    ) throws OrtException {
        List<String[]> result = new ArrayList<>();

        for (String imagePath : imagePaths) {
            Mat image = Imgcodecs.imread(imagePath);
            Mat processedImage = preprocessImage(image, modelHeight, keepRatioWithPad, adjustContrast);

            int batchSize = 1;
            long[] shape = new long[]{batchSize, 3, processedImage.height(), processedImage.width()};
            OnnxTensor imageOnnxTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), FloatBuffer.wrap(toFloatArray(processedImage)), shape);

            long[] lengthForPred = new long[]{batchMaxLength};
            OnnxTensor lengthForPredTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), lengthForPred);

            long[] textForPred = new long[batchMaxLength + 1];
            OnnxTensor textForPredTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), textForPred);

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("image", imageOnnxTensor);
            inputs.put("text", textForPredTensor);
            inputs.put("length", lengthForPredTensor);
            OrtSession.Result preds = session.run(inputs);
            float[][][] predsArray = (float[][][]) preds.get(0).getValue();

            // Apply softmax and filter ignore_idx
            float[][][] predsProb = softmax(predsArray);
            for (int i = 0; i < predsProb.length; i++) {
                for (int j = 0; j < predsProb[i].length; j++) {
                    predsProb[i][j][ignoreIdx] = 0;
                }
            }

            predsProb = normalize(predsProb);

            int[][] predsIndex = argmax(predsProb);
            int[] predsSize = new int[]{predsIndex[0].length};
            String[] predsStr = converter.decodeGreedy(flatten(predsIndex), predsSize).toArray(new String[0]);

            for (String pred : predsStr) {
                float[] maxProbs = maxProb(predsProb);
                float confidenceScore = mean(maxProbs);
                result.add(new String[]{pred, String.valueOf(confidenceScore)});
            }
        }

        return result;
    }

    private static float[] toFloatArray(Mat image) {
        float[] floatArray = new float[(int) (image.total() * image.channels())];
        image.get(0, 0, floatArray);
        return floatArray;
    }

    private static float[][][] softmax(float[][][] x) {
        float[][][] result = new float[x.length][x[0].length][x[0][0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                float max = Float.NEGATIVE_INFINITY;
                for (int k = 0; k < x[i][j].length; k++) {
                    if (x[i][j][k] > max) {
                        max = x[i][j][k];
                    }
                }
                float sum = 0;
                for (int k = 0; k < x[i][j].length; k++) {
                    result[i][j][k] = (float) Math.exp(x[i][j][k] - max);
                    sum += result[i][j][k];
                }
                for (int k = 0; k < x[i][j].length; k++) {
                    result[i][j][k] /= sum;
                }
            }
        }
        return result;
    }

    private static float[][][] normalize(float[][][] x) {
        float[][][] result = new float[x.length][x[0].length][x[0][0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                float sum = 0;
                for (int k = 0; k < x[i][j].length; k++) {
                    sum += x[i][j][k];
                }
                for (int k = 0; k < x[i][j].length; k++) {
                    result[i][j][k] = x[i][j][k] / sum;
                }
            }
        }
        return result;
    }

    private static int[][] argmax(float[][][] x) {
        int[][] result = new int[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                int maxIndex = -1;
                float maxValue = Float.NEGATIVE_INFINITY;
                for (int k = 0; k < x[i][j].length; k++) {
                    if (x[i][j][k] > maxValue) {
                        maxValue = x[i][j][k];
                        maxIndex = k;
                    }
                }
                result[i][j] = maxIndex;
            }
        }
        return result;
    }

    private static int[] flatten(int[][] x) {
        int[] result = new int[x.length * x[0].length];
        int index = 0;
        for (int[] row : x) {
            for (int value : row) {
                result[index++] = value;
            }
        }
        return result;
    }

    private static float[] maxProb(float[][][] x) {
        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < x[i].length; j++) {
                for (int k = 0; k < x[i][j].length; k++) {
                    if (x[i][j][k] > max) {
                        max = x[i][j][k];
                    }
                }
            }
            result[i] = max;
        }
        return result;
    }

    private static float mean(float[] x) {
        float sum = 0;
        for (float value : x) {
            sum += value;
        }
        return sum / x.length;
    }

    public static void main(String[] args) throws OrtException {
        // Load model and initialize ONNX runtime
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = env.createSession("crnn.onnx", options);

        // Example usage
        List<String> testImagePaths = new ArrayList<>();
        testImagePaths.add("english.png");

        // Example parameters for preprocessing
        int modelHeight = 32;  // Example model height
        boolean keepRatioWithPad = false;  // Example flag for
        double adjustContrast = 0; // Example contrast adjustment factor
        Map<String, String> dictList = new HashMap<>();
        dictList.put("en", "en.txt");
        CTC converter = new CTC("0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", new HashMap<>(), dictList);

        List<String[]> results = getText(session, converter, testImagePaths, 25, -1, "greedy", 5, "cpu", modelHeight, keepRatioWithPad, adjustContrast);
        for (String[] result : results) {
            System.out.println("Prediction: " + result[0] + ", Confidence: " + result[1]);
        }
    }
}
