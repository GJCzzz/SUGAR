package com.example.myapplication;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class getImage {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static class BoxImagePair {
        public List<Point> box;
        public Mat image;

        public BoxImagePair(List<Point> box, Mat image) {
            this.box = box;
            this.image = image;
        }
    }

    public static List<BoxImagePair> getImageList(
            List<List<Point>> horizontalList,
            List<List<Point>> freeList,
            Mat img,
            int modelHeight,
            boolean sortOutput) {

        List<BoxImagePair> imageList = new ArrayList<>();
        double maxRatioHori = 1;
        double maxRatioFree = 1;
        int maximumY = img.rows();
        int maximumX = img.cols();

        for (List<Point> box : freeList) {
            Mat transformedImg = fourPointTransform(img, box);
            double ratio = calculateRatio(transformedImg.cols(), transformedImg.rows());
            int newWidth = (int) (modelHeight * ratio);
            if (newWidth > 0) {
                Mat cropImg = computeRatioAndResize(transformedImg, transformedImg.cols(), transformedImg.rows(), modelHeight);
                imageList.add(new BoxImagePair(box, cropImg));
                maxRatioFree = Math.max(ratio, maxRatioFree);
            }
        }

        maxRatioFree = Math.ceil(maxRatioFree);

        for (List<Point> box : horizontalList) {
            int xMin = Math.max(0, (int) box.get(0).x);
            int xMax = Math.min((int) box.get(1).x, maximumX);
            int yMin = Math.max(0, (int) box.get(2).y);
            int yMax = Math.min((int) box.get(3).y, maximumY);

            Mat cropImg = img.submat(yMin, yMax, xMin, xMax);
            int width = xMax - xMin;
            int height = yMax - yMin;
            double ratio = calculateRatio(width, height);
            int newWidth = (int) (modelHeight * ratio);

            if (newWidth > 0) {
                cropImg = computeRatioAndResize(cropImg, width, height, modelHeight);
                imageList.add(new BoxImagePair(box, cropImg));
                maxRatioHori = Math.max(ratio, maxRatioHori);
            }
        }

        maxRatioHori = Math.ceil(maxRatioHori);
        double maxRatio = Math.max(maxRatioHori, maxRatioFree);
        int maxWidth = (int) (Math.ceil(maxRatio) * modelHeight);

        if (sortOutput) {
            imageList = imageList.stream()
                    .sorted(Comparator.comparingInt(item -> (int) item.box.get(0).y))
                    .collect(Collectors.toList());
        }

        return imageList;
    }

    public static Mat computeRatioAndResize(Mat img, int width, int height, int modelHeight) {
        double ratio = calculateRatio(width, height);
        int newWidth;
        if (ratio < 1.0) {
            ratio = calculateRatio(width, height);
            newWidth = modelHeight;
            int newHeight = (int) (modelHeight * ratio);
            Imgproc.resize(img, img, new Size(newWidth, newHeight), 0, 0, Imgproc.INTER_LANCZOS4);
        } else {
            newWidth = (int) (modelHeight * ratio);
            Imgproc.resize(img, img, new Size(newWidth, modelHeight), 0, 0, Imgproc.INTER_LANCZOS4);
        }
        return img;
    }

    public static Mat fourPointTransform(Mat image, List<Point> pts) {
        Point tl = pts.get(0);
        Point tr = pts.get(1);
        Point br = pts.get(2);
        Point bl = pts.get(3);

        double widthA = Math.sqrt(Math.pow(br.x - bl.x, 2) + Math.pow(br.y - bl.y, 2));
        double widthB = Math.sqrt(Math.pow(tr.x - tl.x, 2) + Math.pow(tr.y - tl.y, 2));
        int maxWidth = (int) Math.max(widthA, widthB);

        double heightA = Math.sqrt(Math.pow(tr.x - br.x, 2) + Math.pow(tr.y - br.y, 2));
        double heightB = Math.sqrt(Math.pow(tl.x - bl.x, 2) + Math.pow(tl.y - bl.y, 2));
        int maxHeight = (int) Math.max(heightA, heightB);

        Mat dst = Mat.zeros(4, 2, CvType.CV_32F);
        dst.put(0, 0, 0.0, 0.0, maxWidth - 1.0, 0.0, maxWidth - 1.0, maxHeight - 1.0, 0.0, maxHeight - 1.0);

        Mat src = new Mat(4, 2, CvType.CV_32F);
        src.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y);

        Mat M = Imgproc.getPerspectiveTransform(src, dst);
        Mat warped = new Mat();
        Imgproc.warpPerspective(image, warped, M, new Size(maxWidth, maxHeight));

        return warped;
    }

    public static double calculateRatio(int width, int height) {
        double ratio = (double) width / height;
        if (ratio < 1.0) {
            ratio = 1.0 / ratio;
        }
        return ratio;
    }


}
