package com.example.myapplication;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE_REQUEST = 1;
    private OrtEnvironment ortEnvironment;
    private OrtSession ortSessionDetector;
    private OrtSession ortSessionRecognizer;
    private static final String DETECTOR_MODEL_FILE_NAME = "detector_craft.onnx";
    private static final String RECOGNIZER_MODEL_FILE_NAME = "crnn.onnx";

    private ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        requestPermissions();

        Button buttonOpenGallery = findViewById(R.id.button_open_gallery);
        buttonOpenGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openGallery();
            }
        });

        imageView = findViewById(R.id.imageView);

        // Load detector model
        try {
            String detectorModelPath = loadModelFile(DETECTOR_MODEL_FILE_NAME);
            ortEnvironment = OrtEnvironment.getEnvironment();
            ortSessionDetector = ortEnvironment.createSession(detectorModelPath, new OrtSession.SessionOptions());
            Log.d("MainActivity", "Detector model loaded successfully");
        } catch (OrtException | IOException e) {
            Log.e("MainActivity", "Failed to load the detector model", e);
            Toast.makeText(this, "Failed to load the detector model", Toast.LENGTH_SHORT).show();
        }

        // Load recognizer model
        try {
            String recognizerModelPath = loadModelFile(RECOGNIZER_MODEL_FILE_NAME);
            ortSessionRecognizer = ortEnvironment.createSession(recognizerModelPath, new OrtSession.SessionOptions());
            Log.d("MainActivity", "Recognizer model loaded successfully");
        } catch (OrtException | IOException e) {
            Log.e("MainActivity", "Failed to load the recognizer model", e);
            Toast.makeText(this, "Failed to load the recognizer model", Toast.LENGTH_SHORT).show();
        }

        // Initialize OpenCV
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV initialization failed", Toast.LENGTH_SHORT).show();
        } else {
            Log.d("MainActivity", "OpenCV initialization succeeded");
        }
    }

    private void requestPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, PICK_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            Uri uri = data.getData();
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);

                // Convert Bitmap to OpenCV Mat
                Mat mat = new Mat();
                Utils.bitmapToMat(bitmap, mat);

                // Perform text recognition
                List<String[]> textResults = performTextRecognition(bitmap);

                // Display or process text recognition results
                // Example: Log results
                for (String[] result : textResults) {
                    Log.d("TextRecognition", "Prediction: " + result[0] + ", Confidence: " + result[1]);
                }

            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(this, "Failed to open image", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private List<String[]> performTextRecognition(Bitmap bitmap) {
        List<String[]> results = new ArrayList<>();
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);

        try {
            // Example parameters for text recognition
            int modelHeight = 32;  // Example model height
            boolean keepRatioWithPad = false;  // Example flag for
            double adjustContrast = 0; // Example contrast adjustment factor
            CTC converter = new CTC("0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", new HashMap<>(), Collections.emptyMap());
            FinalGet finalGet = new FinalGet();
            // Process the selected image for text detection
            Pair<List<List<Rect>>, List<List<MatOfPoint2f>>> detectionResult = finalGet.detect(
                    DETECTOR_MODEL_FILE_NAME, ortSessionDetector, mat, 20, 0.7f, 0.4f, 0.4f, 2560, 1.0f,
                    0.1, 0.5, 0.5, 0.5, 1.0, true, null, 0.2f, 0.2f, 3, 0);

            // Process each detected region
            List<List<Rect>> horizontalListAgg = detectionResult.first;
            List<List<MatOfPoint2f>> freeListAgg = detectionResult.second;

            for (List<Rect> rectList : horizontalListAgg) {
                for (Rect rect : rectList) {
                    Bitmap croppedBitmap = Bitmap.createBitmap(bitmap, rect.x, rect.y, rect.width, rect.height);

                    Mat croppedMat = new Mat();
                    Utils.bitmapToMat(croppedBitmap, croppedMat);

                    // Perform text recognition
                    List<String[]> textResults = Recognizer.getText(
                            ortSessionRecognizer, converter, Collections.singletonList(String.valueOf(croppedMat)), 25, -1, "greedy", 5, "cpu",
                            modelHeight, keepRatioWithPad, adjustContrast);

                    // Collect results
                    results.addAll(textResults);
                }
            }

        } catch (OrtException e) {
            e.printStackTrace();
            Toast.makeText(this, "Failed to perform text recognition", Toast.LENGTH_SHORT).show();
        }

        return results;
    }

    private String loadModelFile(String fileName) throws IOException {
        File file = new File(getFilesDir(), fileName);
        if (!file.exists()) {
            try (InputStream is = getAssets().open(fileName);
                 FileOutputStream fos = new FileOutputStream(file)) {
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, bytesRead);
                }
            }
        }
        Log.d("MainActivity", "Model file path: " + file.getAbsolutePath());
        return file.getAbsolutePath();
    }
}
