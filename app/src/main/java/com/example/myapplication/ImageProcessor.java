package com.example.myapplication;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class ImageProcessor {

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

    private static class NormalizePAD {
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

    private static class AlignCollate {
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

        public Mat[] preprocessBatch(List<Mat> batch) {
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

    public static List<BoxImagePair> processImages(
            List<List<Point>> horizontalList,
            List<List<Point>> freeList,
            Mat img,
            int modelHeight,
            boolean sortOutput,
            int imgH,
            int imgW,
            boolean keepRatioWithPad,
            double adjustContrast
    ) {
        List<BoxImagePair> imageList = new ArrayList<>();
        double maxRatioHori = 1;
        double maxRatioFree = 1;
        int maximumY = img.rows();
        int maximumX = img.cols();

        AlignCollate alignCollate = new AlignCollate(imgH, imgW, keepRatioWithPad, adjustContrast);

        // Process horizontal lists
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
                Mat[] processedImages = alignCollate.preprocessBatch(Arrays.asList(cropImg));
                if (processedImages.length > 0) {
                    imageList.add(new BoxImagePair(box, processedImages[0]));
                    maxRatioHori = Math.max(ratio, maxRatioHori);
                }
            }
        }

        // Sort output if required
        if (sortOutput) {
            imageList = imageList.stream()
                    .sorted(Comparator.comparingInt(item -> (int) item.box.get(0).y))
                    .collect(Collectors.toList());
        }

        return imageList;
    }

    private static Mat computeRatioAndResize(Mat img, int width, int height, int modelHeight) {
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

    private static Mat fourPointTransform(Mat image, List<Point> pts) {
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

    private static double calculateRatio(int width, int height) {
        double ratio = (double) width / height;
        if (ratio < 1.0) {
            ratio = 1.0 / ratio;
        }
        return ratio;
    }
}
