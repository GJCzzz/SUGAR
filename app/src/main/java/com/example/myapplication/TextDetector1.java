package com.example.myapplication;

import android.util.Log;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.*;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.*;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class TextDetector1 {
    public final static String TAG = "TextDetector";

    public static float[][][][] testNet(Mat[] images, OrtEnvironment ortEnv, OrtSession ortSession) throws OrtException {
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
        float[][][][] outputData = new float[0][][][];
        try {
            long[] shape = new long[]{1, 3, 608, 800};
            OnnxTensor tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(imgData), shape);
            Map<String, OnnxTensor> inputs = Collections.singletonMap("input", tensor);
            OrtSession.Result results = ortSession.run(inputs);
            Log.w(TAG, "testNet: runing sucessfully" );
            outputData = (float[][][][]) results.get(0).getValue();
        } catch (OrtException e) {
            e.printStackTrace();
        }
        ortSession.close();
        ortEnv.close();
        return outputData;
    }
}
