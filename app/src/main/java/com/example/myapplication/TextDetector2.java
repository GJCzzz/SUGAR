package com.example.myapplication;

import android.util.Log;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class TextDetector2 {
    public final static String TAG = "TextDetector";

    public static float[][][][] testNet(Mat[] images) {
        // 加载模型
        OrtEnvironment ortEnv = null;
        OrtSession ortSession = null;
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            OrtSession session = env.createSession("detector_craft.onnx", options);
        } catch (OrtException e) {
            e.printStackTrace();
        }

        // 图像预处理和推理
        float[][][][] outputData = new float[0][][][];
        if (ortEnv != null && ortSession != null) {
            Mat image = images[0];
            Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB);
            Imgproc.resize(image, image, new Size(800, 608));
            image.convertTo(image, CvType.CV_32F);
            FloatBuffer buffer = FloatBuffer.allocate(3 * 800 * 608);
            image.get(0, 0, buffer.array());
            float[] imgData = new float[3 * 800 * 608];
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < 608; h++) {
                    for (int w = 0; w < 800; w++) {
                        imgData[c * 608 * 800 + h * 800 + w] = buffer.get((h * 800 + w) * 3 + c);
                    }
                }
            }
            try {
                long[] shape = new long[]{1, 3, 608, 800};
                OnnxTensor tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(imgData), shape);
                Map<String, OnnxTensor> inputs = Collections.singletonMap("input", tensor);
                OrtSession.Result results = ortSession.run(inputs);
                Log.w(TAG, "testNet: running successfully");
                outputData = (float[][][][]) results.get(0).getValue();
            } catch (OrtException e) {
                e.printStackTrace();
            }
        }

        // 关闭会话和环境
        if (ortSession != null) {
            try {
                ortSession.close();
            } catch (OrtException e) {
                e.printStackTrace();
            }
        }
        if (ortEnv != null) {
            ortEnv.close();
        }

        return outputData;
    }
}
