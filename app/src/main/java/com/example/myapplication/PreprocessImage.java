package com.example.myapplication;

import static com.example.myapplication.GetBoxes.adjustResultCoordinates;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.content.Intent;
import android.content.pm.PackageManager;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

import org.opencv.imgproc.Imgproc;

import org.opencv.utils.Converters;


import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;


import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import android.util.Log;
import android.util.Pair;
import java.util.HashMap;
public class PreprocessImage {

    public static Bitmap loadImage(String imgFile) {
        Bitmap img = BitmapFactory.decodeFile(imgFile);
        if (img.getConfig() == Bitmap.Config.ALPHA_8) {
            img = Bitmap.createBitmap(img.getWidth(), img.getHeight(), Bitmap.Config.ARGB_8888);
        }
        return img;
    }

    public static Bitmap toGrayscale(Bitmap bmpOriginal) {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
    }

    public static Pair<Bitmap, Bitmap> reformatInput(Object image) {
        Bitmap img = null;
        Bitmap imgCvGrey = null;

        if (image instanceof String) {
            String imagePath = (String) image;
            imgCvGrey = BitmapFactory.decodeFile(imagePath, new BitmapFactory.Options());
            img = loadImage(imagePath);
        } else if (image instanceof byte[]) {
            byte[] byteArray = (byte[]) image;
            img = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
            imgCvGrey = toGrayscale(img);
        } else if (image instanceof Bitmap) {
            img = (Bitmap) image;
            imgCvGrey = toGrayscale(img);
        }



        return new Pair<>(img, imgCvGrey);
    }
}
//prerocess


 class GetBoxes {

    public static class Triple<T1, T2, T3> {
        public T1 first;
        public T2 second;
        public T3 third;

        public Triple(T1 first, T2 second, T3 third) {
            this.first = first;
            this.second = second;
            this.third = third;
        }
    }

    public static Triple<List<MatOfPoint2f>, Mat, List<Integer>> getDetBoxesCore(Mat textmap, Mat linkmap, double textThreshold, double linkThreshold, double lowText, boolean estimateNumChars) {
        linkmap = linkmap.clone();
        textmap = textmap.clone();
        int img_h = textmap.rows();
        int img_w = textmap.cols();

        Mat text_score = new Mat();
        Imgproc.threshold(textmap, text_score, lowText, 1, Imgproc.THRESH_BINARY);
        Mat link_score = new Mat();
        Imgproc.threshold(linkmap, link_score, linkThreshold, 1, Imgproc.THRESH_BINARY);

        Mat text_score_comb = new Mat();
        Core.add(text_score, link_score, text_score_comb);
        Core.min(text_score_comb, new Scalar(1), text_score_comb);

        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int nLabels = Imgproc.connectedComponentsWithStats(text_score_comb, labels, stats, centroids, 4, CvType.CV_32S);

        List<MatOfPoint2f> det = new ArrayList<>();
        List<Integer> mapper = new ArrayList<>();

        for (int k = 1; k < nLabels; k++) {
            int size = (int) stats.get(k, Imgproc.CC_STAT_AREA)[0];
            if (size < 10) continue;



// 1. 创建一个临时的 Mat 对象，用于存储 labels == k 的结果
            Mat mask = new Mat();
            Core.compare(labels, new Scalar(k), mask, Core.CMP_EQ);

// 2. 通过 mask 来获取 textmap 中符合条件的最大值
            Mat textmap_masked = new Mat();
            textmap.copyTo(textmap_masked, mask);

            Scalar maxVal = Scalar.all(Core.minMaxLoc(textmap_masked).maxVal);

// 3. 比较最大值是否小于阈值，如果是则跳过当前循环
            if (maxVal.val[0] <  textThreshold) {
                continue; // 根据你的逻辑继续下一个循环或操作
            }

// 获取 maxVal 的值并进行比较


            Mat segmap = Mat.zeros(textmap.size(), CvType.CV_8UC1);
            Core.compare(labels, new Scalar(k), segmap, Core.CMP_EQ);

            if (estimateNumChars) {
                Mat character_locs = new Mat();
                Mat textmap_clone = textmap.clone();
                Core.subtract(textmap_clone, linkmap, textmap_clone);

                // 执行乘法操作
                Core.multiply(segmap, textmap_clone, character_locs);
                Imgproc.threshold(character_locs, character_locs, textThreshold, 1, Imgproc.THRESH_BINARY);
                Mat character_labels = new Mat();
                Imgproc.connectedComponents(character_locs, character_labels, 4, CvType.CV_32S);
                mapper.add((int) Core.minMaxLoc(character_labels).maxVal);
            } else {
                mapper.add(k);
            }

            Core.compare(link_score, new Scalar(1), segmap, Core.CMP_NE);

            int x = (int) stats.get(k, Imgproc.CC_STAT_LEFT)[0];
            int y = (int) stats.get(k, Imgproc.CC_STAT_TOP)[0];
            int w = (int) stats.get(k, Imgproc.CC_STAT_WIDTH)[0];
            int h = (int) stats.get(k, Imgproc.CC_STAT_HEIGHT)[0];

            int niter = (int) Math.sqrt(size * Math.min(w, h) / (double) (w * h)) * 2;

            int sx = Math.max(0, x - niter);
            int sy = Math.max(0, y - niter);
            int ex = Math.min(img_w, x + w + niter + 1);
            int ey = Math.min(img_h, y + h + niter + 1);

            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1 + niter, 1 + niter));
            Mat dilated = new Mat();
            Imgproc.dilate(segmap.submat(sy, ey, sx, ex), dilated, kernel);
            dilated.copyTo(segmap.submat(sy, ey, sx, ex));

            MatOfPoint2f points = new MatOfPoint2f();
            Core.findNonZero(segmap, points);

            RotatedRect box = Imgproc.minAreaRect(points);

            if (Math.abs(1 - Math.max(box.size.width, box.size.height) / Math.min(box.size.width, box.size.height)) <= 0.1) {
                box.points(points.toArray());
                List<Point> pointsList = points.toList();
                pointsList.sort((p1, p2) -> Double.compare(p1.x + p1.y, p2.x + p2.y));
                det.add(new MatOfPoint2f(pointsList.toArray(new Point[0])));
            } else {
                det.add(points);
            }
        }

        return new Triple<>(det, labels,mapper);
    }

//    getDetBoxes_core function


    static class Point3 {
        public double x, y, z;

        public Point3(double x, double y, double z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }
    }

    // Helper class for 4D points
    static class Point4 {
        public double x, y, z, w;

        public Point4(double x, double y, double z, double w) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }
    }




    public static Point warpCoord(Mat Minv, Point pt) {
        Mat ptMat = new Mat(3, 1, CvType.CV_64F);
        ptMat.put(0, 0, pt.x);
        ptMat.put(1, 0, pt.y);
        ptMat.put(2, 0, 1);

        Mat result = Minv.mul(ptMat);
        double[] data = result.get(0, 0);
        return new Point(data[0] / data[2], data[1] / data[2]);
    }



    public static List<MatOfPoint2f> getPolyCore(List<MatOfPoint2f> boxes, Mat labels, List<Integer> mapper, Mat linkmap) {
        int num_cp = 5;
        double max_len_ratio = 0.7;
        double expand_ratio = 1.45;
        double max_r = 2.0;
        double step_r = 0.2;

        List<MatOfPoint2f> polys = new ArrayList<>();
        for (int k = 0; k < boxes.size(); k++) {
            MatOfPoint2f box = boxes.get(k);
            double w = euclideanDist(box.toArray()[0], box.toArray()[1]);
            double h = euclideanDist(box.toArray()[1], box.toArray()[2]);
            if (w < 10 || h < 10) {
                polys.add(null);
                continue;
            }

            List<Point> tar = Arrays.asList(new Point(0, 0), new Point(w, 0), new Point(w, h), new Point(0, h));
            Mat M = getPerspectiveTransform(box.toList(), tar);
            Mat wordLabel = warpPerspective(labels, M, new Size(w, h));
            Mat Minv;
            try {
                Minv = invertMatrix(M);
            } catch (Exception e) {
                polys.add(null);
                continue;
            }

            int curLabel = mapper.get(k);
            Core.compare(wordLabel, new Scalar(curLabel), wordLabel, Core.CMP_EQ);

            List<Point3> cp = new ArrayList<>();
            double max_len = -1;
            for (int i = 0; i < w; i++) {
                Mat col = wordLabel.col(i);
                Core.MinMaxLocResult mmr = Core.minMaxLoc(col);
                double min = mmr.minVal;
                double max = mmr.maxVal;
                if (min == max) continue;
                double sy = mmr.minLoc.y;
                double ey = mmr.maxLoc.y;
                cp.add(new Point3(i, sy, ey));
                double length = ey - sy + 1;
                if (length > max_len) max_len = length;
            }

            if (h * max_len_ratio < max_len) {
                polys.add(null);
                continue;
            }

            int tot_seg = num_cp * 2 + 1;
            double seg_w = w / tot_seg;
            Point[] pp = new Point[num_cp];
            Point3[] cp_section = new Point3[tot_seg];
            double[] seg_height = new double[num_cp];
            int seg_num = 0;
            int num_sec = 0;
            double prev_h = -1;
            for (int i = 0; i < cp.size(); i++) {
                Point3 p = cp.get(i);
                if ((seg_num + 1) * seg_w <= p.x && seg_num <= tot_seg) {
                    if (num_sec == 0) break;
                    cp_section[seg_num] = new Point3(cp_section[seg_num].x / num_sec, cp_section[seg_num].y / num_sec, cp_section[seg_num].z / num_sec);
                    num_sec = 0;
                    seg_num++;
                    prev_h = -1;
                }
                double cy = (p.y + p.z) * 0.5;
                double cur_h = p.z - p.y + 1;
                cp_section[seg_num] = new Point3(cp_section[seg_num].x + p.x, cp_section[seg_num].y + cy, cp_section[seg_num].z);
                num_sec++;
                if (seg_num % 2 == 0) continue;
                if (prev_h < cur_h) {
                    pp[(seg_num - 1) / 2] = new Point(p.x, cy);
                    seg_height[(seg_num - 1) / 2] = cur_h;
                    prev_h = cur_h;
                }
            }
            if (num_sec != 0) {
                cp_section[tot_seg - 1] = new Point3(cp_section[tot_seg - 1].x / num_sec, cp_section[tot_seg - 1].y / num_sec, cp_section[tot_seg - 1].z / num_sec);
            }
            if (Arrays.asList(pp).contains(null) || seg_w < Arrays.stream(seg_height).max().getAsDouble() * 0.25) {
                polys.add(null);
                continue;
            }

            double half_char_h = Arrays.stream(seg_height).average().orElse(0) * expand_ratio / 2;
            List<Point4> new_pp = new ArrayList<>();
            for (Point p : pp) {
                double dx = cp_section[(Arrays.asList(pp).indexOf(p) * 2 + 2)].x - cp_section[(Arrays.asList(pp).indexOf(p) * 2)].x;
                double dy = cp_section[(Arrays.asList(pp).indexOf(p) * 2 + 2)].y - cp_section[(Arrays.asList(pp).indexOf(p) * 2)].y;
                if (dx == 0) {
                    new_pp.add(new Point4(p.x, p.y - half_char_h, p.x, p.y + half_char_h));
                    continue;
                }
                double rad = -Math.atan2(dy, dx);
                double c = half_char_h * Math.cos(rad);
                double s = half_char_h * Math.sin(rad);
                new_pp.add(new Point4(p.x - s, p.y - c, p.x + s, p.y + c));
            }

            boolean isSppFound = false, isEppFound = false;
            Point4 spp = null, epp = null;
            double grad_s = (pp[1].y - pp[0].y) / (pp[1].x - pp[0].x) + (pp[2].y - pp[1].y) / (pp[2].x - pp[1].x);
            double grad_e = (pp[pp.length - 2].y - pp[pp.length - 1].y) / (pp[pp.length - 2].x - pp[pp.length - 1].x) + (pp[pp.length - 3].y - pp[pp.length - 2].y) / (pp[pp.length - 3].x - pp[pp.length - 2].x);
            for (double r = 0.5; r < max_r; r += step_r) {
                double dx = 2 * half_char_h * r;
                if (!isSppFound) {
                    Mat lineImg = Mat.zeros(wordLabel.size(), CvType.CV_8U);
                    double dy = grad_s * dx;
                    Point4 p = new Point4(new_pp.get(0).x - dx, new_pp.get(0).y - dy, new_pp.get(0).x - dx, new_pp.get(0).y - dy);
                    Imgproc.line(lineImg, new Point(p.x, p.y), new Point(p.z, p.w), new Scalar(1), 1);
                    Mat result = new Mat();
                    Core.bitwise_and(wordLabel, lineImg, result);
                    if (Core.countNonZero(result) == 0 || r + 2 * step_r >= max_r) {
                        spp = p;
                        isSppFound = true;
                    }

                }
                if (!isEppFound) {
                    Mat lineImg = Mat.zeros(wordLabel.size(), CvType.CV_8U);
                    double dy = grad_e * dx;
                    Point4 p = new Point4(new_pp.get(new_pp.size() - 1).x + dx, new_pp.get(new_pp.size() - 1).y + dy, new_pp.get(new_pp.size() - 1).x + dx, new_pp.get(new_pp.size() - 1).y + dy);
                    Imgproc.line(lineImg, new Point(p.x, p.y), new Point(p.z, p.w), new Scalar(1), 1);
                    Mat result1 = new Mat();
                    Core.bitwise_and(wordLabel, lineImg, result1);
                    if (Core.countNonZero(result1) == 0 || r + 2 * step_r >= max_r) {
                        epp = p;
                        isEppFound = true;
                    }

                }
                if (isSppFound && isEppFound) break;
            }

            if (!isSppFound || !isEppFound) {
                polys.add(null);
                continue;
            }

            List<Point> poly = new ArrayList<>();
            poly.add(warpCoord(Minv, new Point(spp.x, spp.y)));
            for (Point4 p : new_pp) {
                poly.add(warpCoord(Minv, new Point(p.x, p.y)));
            }
            poly.add(warpCoord(Minv, new Point(epp.x, epp.y)));
            poly.add(warpCoord(Minv, new Point(epp.z, epp.w)));
            for (int i = new_pp.size() - 1; i >= 0; i--) {
                poly.add(warpCoord(Minv, new Point(new_pp.get(i).z, new_pp.get(i).w)));
            }
            poly.add(warpCoord(Minv, new Point(spp.z, spp.w)));

            polys.add(new MatOfPoint2f(poly.toArray(new Point[0])));
        }
        return polys;
    }

    // Other helper methods





//  getPoly_core function

    public static Mat getPerspectiveTransform(List<Point> box, List<Point> tar) {
        Mat srcMat = Converters.vector_Point2f_to_Mat(box);
        Mat dstMat = Converters.vector_Point2f_to_Mat(tar);
        return Imgproc.getPerspectiveTransform(srcMat, dstMat);
    }

    public static Mat warpPerspective(Mat src, Mat M, Size size) {
        Mat dst = new Mat(size, src.type());
        Imgproc.warpPerspective(src, dst, M, size, Imgproc.INTER_NEAREST);
        return dst;
    }

    public static Mat invertMatrix(Mat M) {
        Mat Minv = new Mat();
        Core.invert(M, Minv);
        return Minv;
    }

    public static double euclideanDist(Point a, Point b) {
        return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
    }

    public static Result getDetBoxes(Mat textmap, Mat linkmap, float textThreshold, float linkThreshold, float lowText, boolean poly, boolean estimateNumChars) {
        if (poly && estimateNumChars) {
            throw new IllegalArgumentException("Estimating the number of characters not currently supported with poly.");
        }

        Triple<List<MatOfPoint2f>, Mat, List<Integer>> detBoxesCoreResult = getDetBoxesCore(textmap, linkmap, textThreshold, linkThreshold, lowText, estimateNumChars);
        List<MatOfPoint2f> boxes = detBoxesCoreResult.first;
        Mat labels = detBoxesCoreResult.second;
        List<Integer> mapper = detBoxesCoreResult.third;

        List<MatOfPoint2f> polys;
        if (poly) {
            polys = getPolyCore(boxes, labels, mapper, linkmap);
        } else {
            polys = new ArrayList<>(boxes.size());
            for (int i = 0; i < boxes.size(); i++) {
                polys.add(null);
            }
        }

        return new Result(boxes, polys, mapper);
    }

public static class Result {
    public List<MatOfPoint2f> boxes;
    public List<MatOfPoint2f> polys;
    public List<Integer> mapper;

    public Result(List<MatOfPoint2f> boxes, List<MatOfPoint2f> polys, List<Integer> mapper) {
        this.boxes = boxes;
        this.polys = polys;
        this.mapper = mapper;
    }
    public List<MatOfPoint2f> getBoxes() {
        return boxes;
    }

    // 如果需要，可以添加获取 polys 和 mapper 的方法
    public List<MatOfPoint2f> getPolys() {
        return polys;
    }

    public List<Integer> getMapper() {
        return mapper;
    }
}
//    getDetBoxes function

    public static List<MatOfPoint2f> adjustResultCoordinates (List < MatOfPoint2f > polys,
                                                              double ratio_w, double ratio_h, double ratio_net){
        if (polys != null && !polys.isEmpty()) {
            for (int k = 0; k < polys.size(); k++) {
                if (polys.get(k) != null) {
                    Core.multiply(polys.get(k), new Scalar(ratio_w * ratio_net, ratio_h * ratio_net), polys.get(k));
                }
            }
        }
        return polys;
    }
//    adjustResultCoordinates function
}



class ImageUtils {

    public static Mat normalizeMeanVariance(Mat inImg, Scalar mean, Scalar variance) {
        Mat img = new Mat();
        inImg.convertTo(img, CvType.CV_32F);

        // Creating mean and variance matrices
        Mat meanMat = new Mat(img.size(), img.type(), new Scalar(mean.val[0] * 255.0, mean.val[1] * 255.0, mean.val[2] * 255.0));
        Mat varianceMat = new Mat(img.size(), img.type(), new Scalar(variance.val[0] * 255.0, variance.val[1] * 255.0, variance.val[2] * 255.0));

        // Subtracting mean
        Core.subtract(img, meanMat, img);

        // Dividing by variance
        Core.divide(img, varianceMat, img);

        return img;
    }


    public static GetBoxes.Triple<Mat, Float,Size> resizeAspectRatio(Mat img, int squareSize, int interpolation, float magRatio) {
        int height = img.rows();
        int width = img.cols();

        float targetSize = magRatio * Math.max(height, width);

        if (targetSize > squareSize) {
            targetSize = squareSize;
        }

        float ratio = targetSize / Math.max(height, width);

        int targetH = Math.round(height * ratio);
        int targetW = Math.round(width * ratio);
        Mat proc = new Mat();
        Imgproc.resize(img, proc, new Size(targetW, targetH), 0, 0, interpolation);

        int targetH32 = targetH;
        int targetW32 = targetW;
        if (targetH % 32 != 0) {
            targetH32 = targetH + (32 - targetH % 32);
        }
        if (targetW % 32 != 0) {
            targetW32 = targetW + (32 - targetW % 32);
        }
        Mat resized = Mat.zeros(new Size(targetW32, targetH32), img.type());
        proc.copyTo(resized.submat(0, targetH, 0, targetW));

        Size sizeHeatmap = new Size(targetW / 2, targetH / 2);

        return new GetBoxes.Triple<>(resized,ratio, sizeHeatmap);
    }
}
//俩resize


 class TextDetector {

    private OrtEnvironment env;
    private OrtSession session;

    public TextDetector(String modelPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath, new OrtSession.SessionOptions());
    }



    public static float[] matToFloatArray(Mat mat) {
        int rows = mat.rows();
        int cols = mat.cols();
        float[] floatArray = new float[rows * cols * mat.channels()];
        mat.convertTo(mat, CvType.CV_32F);
        mat.get(0, 0, floatArray);
        return floatArray;
    }

    public static OnnxTensor matToFloatTensor(Mat mat) throws OrtException {
        // 转换 Mat 为 float 数组
        float[] floatArray = matToFloatArray(mat);

        // 获取 Mat 的形状
        long[] shape = {1, mat.channels(), mat.rows(), mat.cols()};

        // 创建 ONNX Runtime 的环境
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        // 创建 FloatTensor
        OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(floatArray), shape);

        return tensor;
    }
    private static FloatBuffer listMatToFloatBuffer(List<Mat> mats) {
        // 计算总共需要的空间大小
        int totalSize = 0;
        for (Mat mat : mats) {
            totalSize += mat.rows() * mat.cols() * mat.channels();
        }

        // 创建FloatBuffer
        FloatBuffer floatBuffer = FloatBuffer.allocate(totalSize);

        // 填充数据到FloatBuffer
        for (Mat mat : mats) {
            float[] floatArray = new float[(int) (mat.total() * mat.channels())];
            mat.get(0, 0, floatArray); // 将Mat转换为float数组
            floatBuffer.put(floatArray); // 将float数组放入FloatBuffer
        }

        floatBuffer.flip(); // 切换到读取模式

        return floatBuffer;

    }
    public class BoxMapperPair {
        private MatOfPoint2f box;
        private int mapper;

        public BoxMapperPair(MatOfPoint2f box, int mapper) {
            this.box = box;
            this.mapper = mapper;
        }

        public MatOfPoint2f getBox() {
            return box;
        }

        public int getMapper() {
            return mapper;
        }
    }

    public Pair testNet(            OrtSession session,int canvasSize, float magRatio,
                                          Mat image,  float textThreshold, float linkThreshold, float lowText, boolean poly, boolean estimateNumChars) throws OrtException {
        List<Mat> img_resized_list = new ArrayList<>();

        // resize
        Mat img_resized = new Mat();
        float target_ratio = 0;
        Size size_heatmap = new Size();

        // 假设 resize_aspect_ratio 是一个方法用于按比例调整图像大小
        ImageUtils.resizeAspectRatio(image, canvasSize, Imgproc.INTER_LINEAR, magRatio);

        img_resized_list.add(img_resized);

        float ratio_h = 1 / target_ratio;
        float ratio_w = 1 / target_ratio;

        // preprocessing
        List<Mat> x = new ArrayList<>();
        for (Mat n_img : img_resized_list) {
            Scalar mean = new Scalar(0.485, 0.456, 0.406);
            Scalar variance = new Scalar(0.229, 0.224, 0.225);

            Mat normalized = ImageUtils.normalizeMeanVariance(n_img, mean, variance); // 假设这是一个归一化处理的方法
            Mat transposed = new Mat();
            Core.transpose(normalized, transposed); // 假设这是一个转置矩阵的方法
            x.add(transposed);
        }
        FloatBuffer inputs = listMatToFloatBuffer(x);
        long[] shape = new long[]{x.size(), x.get(0).rows(), x.get(0).cols(), x.get(0).channels()};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputs, shape);
        Map<String, OnnxTensor> input = Collections.singletonMap("input", inputTensor);


//        这里就是部署的地方，y是第一个变量

        OrtSession.Result result = session.run(input);
        float[][][][] y = (float[][][][]) result.get(0).getValue();
        // 获取结果列表，假设结果是一个 Mat 列表
// 假设 y 是 float[][][][] 类型的输出结果
        float[][][] y_ = y[0];
// 提取 score_text 和 score_link
        float[][] scoreText = y_[0]; // 假设 score_text 是第一个通道
        float[][] scoreLink = y_[1]; // 假设 score_link 是第二个通道
        int rows = scoreText.length;
        int cols = scoreText[0].length;
        Mat matScoreText = new Mat(rows, cols, CvType.CV_32FC1); // CV_32FC1 表示单通道浮点型
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matScoreText.put(i, j, scoreText[i][j]);
            }
        }

// 转换 scoreLink 到 Mat
        rows = scoreLink.length;
        cols = scoreLink[0].length;
        Mat matScoreLink = new Mat(rows, cols, CvType.CV_32FC1); // CV_32FC1 表示单通道浮点型
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matScoreLink.put(i, j, scoreLink[i][j]);
            }
        }
// 后处理和坐标调整
// 请根据实际情况实现 getDetBoxes 和 adjustResultCoordinates 方法
        GetBoxes.Result results = GetBoxes.getDetBoxes(matScoreText, matScoreLink, textThreshold, linkThreshold, lowText, poly, estimateNumChars);
        List<MatOfPoint2f> boxes = results.getBoxes();
        List<MatOfPoint2f> polys = results.getPolys();
        List<Integer> mapper = results.getMapper();
        double ratio_net = 2;
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h, ratio_net);
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net);
// 将结果存储到 boxes_list 和 polys_list
        List<List<MatOfPoint2f>> boxesList = new ArrayList<>();
        List<List<MatOfPoint2f>> polysList = new ArrayList<>();

        for (int k = 0; k < polys.size(); k++) {
            // 处理 estimate_num_chars
            if (estimateNumChars) {
                List<BoxMapperPair> boxMapperPairs = new ArrayList<>();

                // 创建 BoxMapperPair 对象并存储到列表中
                for (int i = 0; i < boxes.size(); i++) {
                    boxMapperPairs.add(new BoxMapperPair(boxes.get(i), mapper.get(i)));
                }

                // 更新 boxes 和 mapper
                for (int i = 0; i < boxes.size(); i++) {
                    boxes.set(i, boxMapperPairs.get(i).getBox());
                    mapper.set(i, boxMapperPairs.get(i).getMapper());
                }
            }

            // 处理 polys
            if (polys.get(k) == null) {
                polys.set(k, boxes.get(k)); // 如果 polys[k] 是 null，则使用 boxes[k]
            }
        }

        boxesList.add(new ArrayList<>(boxes));
        polysList.add(new ArrayList<>(polys));
        return new Pair<>(boxesList, polysList);
    }
// 返回结果



     public List<List<MatOfPoint2f>> getTextbox(OrtSession session, Mat image, int canvasSize, float magRatio, float textThreshold,
                                                float linkThreshold, float lowText, boolean poly, boolean optimalNumChars) throws OrtException {

         List<List<MatOfPoint2f>> result = new ArrayList<>();

         Pair<List<List<MatOfPoint2f>>, List<List<MatOfPoint2f>>> boxesAndPolys = testNet(session, canvasSize, magRatio, image, textThreshold, linkThreshold, lowText, poly, optimalNumChars);

         List<List<MatOfPoint2f>> boxesList = boxesAndPolys.first;
         List<List<MatOfPoint2f>> polysList = boxesAndPolys.second;

         for (List<MatOfPoint2f> polys : polysList) {
             List<MatOfPoint2f> singleImgResult = new ArrayList<>();
             for (MatOfPoint2f box : polys) {
                 singleImgResult.add(box);
             }
             result.add(singleImgResult);
         }

         return result;
     }
    }


//得到盒子
//444


 class TextBoxUtils {

//    public static class GroupTextBoxResult {
//        public List<Rect> mergedList;
//        public List<Rect> freeList;
//
//        public GroupTextBoxResult(List<Rect> mergedList, List<Rect> freeList) {
//            this.mergedList = mergedList;
//            this.freeList = freeList;
//        }
//    }

    public static Pair<List<MatOfPoint2f>,List<MatOfPoint2f>> groupTextBox(List<MatOfPoint2f> polys, double slopeThs, double yCenterThs, double heightThs, double widthThs, double addMargin, boolean sortOutput) {
        List<Rect> horizontalList = new ArrayList<>();
        List<Rect> freeList = new ArrayList<>();
        List<List<Rect>> combinedList = new ArrayList<>();
        List<Rect> mergedList = new ArrayList<>();

        for (MatOfPoint2f poly : polys) {
            Point[] points = poly.toArray();

            double slopeUp = (points[3].y - points[1].y) / Math.max(10, (points[2].x - points[0].x));
            double slopeDown = (points[5].y - points[7].y) / Math.max(10, (points[4].x - points[6].x));

            if (Math.max(Math.abs(slopeUp), Math.abs(slopeDown)) < slopeThs) {
                double xMax = Math.max(Math.max(points[0].x, points[2].x), Math.max(points[4].x, points[6].x));
                double xMin = Math.min(Math.min(points[0].x, points[2].x), Math.min(points[4].x, points[6].x));
                double yMax = Math.max(Math.max(points[1].y, points[3].y), Math.max(points[5].y, points[7].y));
                double yMin = Math.min(Math.min(points[1].y, points[3].y), Math.min(points[5].y, points[7].y));

                horizontalList.add(new Rect((int) xMin, (int) yMin, (int) (xMax - xMin), (int) (yMax - yMin)));
            } else {
                double height = Math.hypot(points[6].x - points[0].x, points[7].y - points[1].y);
                double width = Math.hypot(points[2].x - points[0].x, points[3].y - points[1].y);

                double margin = 1.44 * addMargin * Math.min(width, height);

                double theta13 = Math.abs(Math.atan((points[1].y - points[5].y) / Math.max(10, (points[0].x - points[4].x))));
                double theta24 = Math.abs(Math.atan((points[3].y - points[7].y) / Math.max(10, (points[2].x - points[6].x))));

                double x1 = points[0].x - Math.cos(theta13) * margin;
                double y1 = points[0].y - Math.sin(theta13) * margin;
                double x2 = points[2].x + Math.cos(theta24) * margin;
                double y2 = points[2].y - Math.sin(theta24) * margin;
                double x3 = points[4].x + Math.cos(theta13) * margin;
                double y3 = points[4].y + Math.sin(theta13) * margin;
                double x4 = points[6].x - Math.cos(theta24) * margin;
                double y4 = points[6].y + Math.sin(theta24) * margin;

                Point[] rectPoints = new Point[4];
                rectPoints[0] = new Point(x1, y1);
                rectPoints[1] = new Point(x2, y2);
                rectPoints[2] = new Point(x3, y3);
                rectPoints[3] = new Point(x4, y4);
//                freeList.add(new RotatedRect(rectPoints));
            }
        }

        if (sortOutput) {
            horizontalList.sort(Comparator.comparingDouble(rect -> rect.y + rect.height / 2.0));
        }

        List<Rect> newBox = new ArrayList<>();
        for (Rect poly : horizontalList) {
            if (newBox.isEmpty()) {
                newBox.add(poly);
            } else {
                double yCenter = poly.y + poly.height / 2.0;
                double avgHeight = newBox.stream().mapToDouble(r -> r.height).average().orElse(0);
                double avgYCenter = newBox.stream().mapToDouble(r -> r.y + r.height / 2.0).average().orElse(0);

                if (Math.abs(avgYCenter - yCenter) < yCenterThs * avgHeight) {
                    newBox.add(poly);
                } else {
                    combinedList.add(new ArrayList<>(newBox));
                    newBox.clear();
                    newBox.add(poly);
                }
            }
        }
        if (!newBox.isEmpty()) {
            combinedList.add(new ArrayList<>(newBox));
        }

        for (List<Rect> boxes : combinedList) {
            if (boxes.size() == 1) {
                Rect box = boxes.get(0);
                double margin = addMargin * Math.min(box.width, box.height);
                mergedList.add(new Rect((int) (box.x - margin), (int) (box.y - margin), (int) (box.width + 2 * margin), (int) (box.height + 2 * margin)));
            } else {
                boxes.sort(Comparator.comparingInt(rect -> rect.x));
                List<Rect> mergedBox = new ArrayList<>();
                for (Rect box : boxes) {
                    if (mergedBox.isEmpty()) {
                        mergedBox.add(box);
                    } else {
                        Rect lastBox = mergedBox.get(mergedBox.size() - 1);
                        if ((Math.abs(box.height - lastBox.height) < heightThs * lastBox.height) && (box.x - (lastBox.x + lastBox.width) < widthThs * (box.y + box.height - lastBox.y))) {
                            lastBox.width = box.x + box.width - lastBox.x;
                            lastBox.height = Math.max(lastBox.height, box.y + box.height - lastBox.y);
                        } else {
                            mergedBox.add(box);
                        }
                    }
                }
                for (Rect mbox : mergedBox) {
                    double margin = addMargin * Math.min(mbox.width, mbox.height);
                    mergedList.add(new Rect((int) (mbox.x - margin), (int) (mbox.y - margin), (int) (mbox.width + 2 * margin), (int) (mbox.height + 2 * margin)));
                }
            }
        }

        return new Pair(mergedList, freeList);
    }
}
//group_text_box


 class FinalGet {

    private OrtEnvironment env;
    private OrtSession session;



    private static double diff(List<Double> points) {
        double maxVal = Double.MIN_VALUE;
        double minVal = Double.MAX_VALUE;
        for (double point : points) {
            if (point > maxVal) maxVal = point;
            if (point < minVal) minVal = point;
        }
        return maxVal - minVal;
    }
    // Utility method to convert MatOfPoint2f to Rect
    private static Rect convertToRect(MatOfPoint2f mat) {
        Point[] points = mat.toArray();
        double xMin = Double.MAX_VALUE, yMin = Double.MAX_VALUE, xMax = Double.MIN_VALUE, yMax = Double.MIN_VALUE;
        for (Point point : points) {
            xMin = Math.min(xMin, point.x);
            yMin = Math.min(yMin, point.y);
            xMax = Math.max(xMax, point.x);
            yMax = Math.max(yMax, point.y);
        }
        return new Rect((int) xMin, (int) yMin, (int) (xMax - xMin), (int) (yMax - yMin));
    }

    // Utility method to convert List<Rect> to List<MatOfPoint2f>
    private static List<MatOfPoint2f> convertToMatOfPoint2fList(List<Rect> rectList) {
        List<MatOfPoint2f> matList = new ArrayList<>();
        for (Rect rect : rectList) {
            matList.add(convertToMatOfPoint2f(rect));
        }
        return matList;
    }

    // Utility method to convert Rect to MatOfPoint2f
    private static MatOfPoint2f convertToMatOfPoint2f(Rect rect) {
        Point[] points = new Point[4];
        points[0] = new Point(rect.x, rect.y);
        points[1] = new Point(rect.x + rect.width, rect.y);
        points[2] = new Point(rect.x + rect.width, rect.y + rect.height);
        points[3] = new Point(rect.x, rect.y + rect.height);
        return new MatOfPoint2f(points);
    }
    private static List<Rect> convertToRectList2(List<MatOfPoint2f> matList) {
        List<Rect> resultList = new ArrayList<>();
        for (MatOfPoint2f mat : matList) {
            Rect rect = convertToRect(mat);
            resultList.add(rect);
        }
        return resultList;
    }
    // Utility method to calculate difference
    private static List<Rect> convertToRectList(List<Rect> rectList) {
        List<Rect> resultList = new ArrayList<>();
        for (Rect rect : rectList) {
            resultList.add(rect);
        }
        return resultList;
    }
    public Pair<List<List<Rect>>, List<List<MatOfPoint2f>>> detect(String modelPath,OrtSession session,Mat img, int minSize, float textThreshold, float lowText, float linkThreshold, int canvasSize, float magRatio, double slopeThs, double yCenterThs, double heightThs, double widthThs, double addMargin, boolean reformat, Integer optimalNumChars, float threshold, float bboxMinScore, int bboxMinSize, int maxCandidates) throws OrtException {

        if (reformat) {
            Pair<Bitmap, Bitmap> resultPair = PreprocessImage.reformatInput(img);
            Bitmap img_ = resultPair.first;
        }

        List<List<Rect>> horizontalListAgg = new ArrayList<>();
        List<List<MatOfPoint2f>> freeListAgg = new ArrayList<>();

        TextDetector textDetector = new TextDetector(modelPath); // Instantiate TextDetector
        List<List<MatOfPoint2f>>textBoxList = textDetector.getTextbox(session, img, canvasSize, magRatio, textThreshold, linkThreshold, lowText, false, optimalNumChars == null);

        for (List<MatOfPoint2f> textBox : textBoxList) {
            Pair<List<MatOfPoint2f>, List<MatOfPoint2f>> groupedTextBoxes = TextBoxUtils.groupTextBox(textBox, slopeThs, yCenterThs, heightThs, widthThs, addMargin, optimalNumChars == null);

            List<MatOfPoint2f> horizontalList = groupedTextBoxes.first;
            List<MatOfPoint2f> freeList = groupedTextBoxes.second;

            if (minSize > 0) {
                // Filter horizontalList by minSize
                List<Rect> filteredHorizontalList = new ArrayList<>();
                for (MatOfPoint2f mat : horizontalList) {
                    Rect rect = convertToRect(mat);
                    if (Math.max(rect.height, rect.width) > minSize) {
                        filteredHorizontalList.add(rect);
                    }
                }

                // Filter freeList by minSize
                List<MatOfPoint2f> filteredFreeList = new ArrayList<>();
                for (MatOfPoint2f mat : freeList) {
                    Point[] points = mat.toArray();
                    List<Double> xCoords = new ArrayList<>();
                    List<Double> yCoords = new ArrayList<>();
                    for (Point point : points) {
                        xCoords.add(point.x);
                        yCoords.add(point.y);
                    }
                    if (Math.max(diff(xCoords), diff(yCoords)) > minSize) {
                        filteredFreeList.add(mat);
                    }
                }

                horizontalListAgg.add(convertToRectList(filteredHorizontalList));
                freeListAgg.add(filteredFreeList);
            } else {
                // Add original lists if minSize <= 0
                horizontalListAgg.add(convertToRectList2(horizontalList));
                freeListAgg.add(freeList);
            }
        }

        return new Pair<>(horizontalListAgg, freeListAgg);

    }
}