package com.example.myapplication;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


import java.util.ArrayList;
import java.util.List;

public class AlignCollate {

    private int imgH;
    private int imgW;
    private boolean keepRatioWithPad;
    private double adjustContrast;

    public AlignCollate(int imgH, int imgW, boolean keepRatioWithPad, double adjustContrast) {
        this.imgH = imgH;
        this.imgW = imgW;
        this.keepRatioWithPad = keepRatioWithPad;
        this.adjustContrast = adjustContrast;
    }

    public Mat[] call(List<Mat> batch) {
        List<Mat> filteredBatch = new ArrayList<>();
        for (Mat img : batch) {
            if (img != null) {
                filteredBatch.add(img);
            }
        }

        List<Mat> resizedImages = new ArrayList<>();
        NormalizePAD transform = new NormalizePAD(new Size(imgW, imgH));

        for (Mat image : filteredBatch) {
            Size originalSize = image.size();
            int w = (int) originalSize.width;
            int h = (int) originalSize.height;

            // Adjust contrast if needed
            if (adjustContrast > 0) {
                image = adjustContrastGrey(image, adjustContrast);
            }

            float ratio = w / (float) h;
            int resizedW;
            if (Math.ceil(imgH * ratio) > imgW) {
                resizedW = imgW;
            } else {
                resizedW = (int) Math.ceil(imgH * ratio);
            }

            Mat resizedImage = new Mat();
            Imgproc.resize(image, resizedImage, new Size(resizedW, imgH), 0, 0, Imgproc.INTER_CUBIC);
            resizedImages.add(transform.apply(resizedImage));
        }

        return resizedImages.toArray(new Mat[0]);
    }

    private Mat adjustContrastGrey(Mat img, double target) {
        Mat newImg = new Mat();
        img.convertTo(newImg, -1, target, 0);
        return newImg;
    }



}

class NormalizePAD {
    private Size targetSize;

    public NormalizePAD(Size targetSize) {
        this.targetSize = targetSize;
    }

    public Mat apply(Mat img) {
        int top = 0, bottom = 0, left = 0, right = 0;
        Size size = img.size();
        if (size.width < targetSize.width) {
            left = (int) ((targetSize.width - size.width) / 2);
            right = (int) (targetSize.width - size.width - left);
        }
        if (size.height < targetSize.height) {
            top = (int) ((targetSize.height - size.height) / 2);
            bottom = (int) (targetSize.height - size.height - top);
        }

        Mat padded = new Mat();
        Core.copyMakeBorder(img, padded, top, bottom, left, right, Core.BORDER_CONSTANT, Scalar.all(0));
        return padded;
    }
}