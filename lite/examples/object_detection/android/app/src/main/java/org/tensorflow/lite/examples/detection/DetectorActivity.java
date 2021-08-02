/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.AudioManager;
import android.media.ImageReader.OnImageAvailableListener;
import android.media.ToneGenerator;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.zxingcpp.BarcodeReader;
import com.google.zxing.BarcodeFormat;
import com.google.zxing.BinaryBitmap;
import com.google.zxing.DecodeHintType;
import com.google.zxing.MultiFormatReader;
import com.google.zxing.PlanarYUVLuminanceSource;
import com.google.zxing.Result;
import com.google.zxing.common.HybridBinarizer;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Detector;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    // Configuration values for the prepackaged SSD model.
    private static final int TF_OD_API_INPUT_SIZE = 320;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "key_android2.tflite";
    private static final String TF_OD_API_LABELS_FILE = "labelmap2.txt";
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(3840, 2160);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    final private int REQUEST_CODE_ASK_PERMISSIONS = 123;

    private Detector detector;
    private MultiFormatReader readerJava = new MultiFormatReader();

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;
    private Bitmap keydotBitmap = null;
    private int total = 1;
    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;

    private ToneGenerator beeper = new ToneGenerator(AudioManager.STREAM_NOTIFICATION, 50);
    private String lastText = new String();
    private Rect cropRect = new Rect();
    private Integer imageRotation = 0;
    private Integer imageWidth = 0;
    private Integer imageHeight = 0;


    int widthScale = 0;
    int heightScale = 0;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            this,
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing Detector!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Detector could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        widthScale = rgbFrameBitmap.getWidth() / TF_OD_API_INPUT_SIZE;
        heightScale = rgbFrameBitmap.getHeight() / TF_OD_API_INPUT_SIZE;


        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
        requestPermission();
    }

    private void requestPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED ) {
            ActivityCompat
                    .requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CODE_ASK_PERMISSIONS);
        }
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case REQUEST_CODE_ASK_PERMISSIONS:
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // Permission Granted
                    Toast.makeText(this, "Permission Granted", Toast.LENGTH_SHORT)
                            .show();
                } else {
                    // Permission Denied
                    Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT)
                            .show();
                }
                break;
            default:
                super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();
        BarcodeReader readerCpp = new BarcodeReader();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }


        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Detector.Recognition> results = detector.recognizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        Map<DecodeHintType, Object> hints = new HashMap<>();
                        hints.put(DecodeHintType.POSSIBLE_FORMATS, Arrays.asList(BarcodeFormat.DATA_MATRIX));
                        hints.put(DecodeHintType.TRY_HARDER, true);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        final List<Detector.Recognition> mappedRecognitions =
                                new ArrayList<Detector.Recognition>();

                        for (final Detector.Recognition result : results) {
                            final RectF location = result.getLocation();
//                            final Rect rect = new Rect(0, 0, 0, 0);
//                            location.round(rect);


                            if (location != null && result.getConfidence() >= minimumConfidence) {

                                int left = Math.round(location.left * widthScale);
                                int top = Math.round(location.top * heightScale);
                                int right = Math.round(location.right * widthScale);
                                int bottom = Math.round(location.bottom * heightScale);

//                                keydotBitmap = Bitmap.createBitmap(cropCopyBitmap, Math.max(0, rect.left - 20), Math.max(0, rect.top - 10), Math.min(cropCopyBitmap.getWidth(), rect.width() + 40), Math.min(cropCopyBitmap.getHeight(), rect.height() + 20));
                                keydotBitmap = Bitmap.createBitmap(rgbFrameBitmap, left, top, right - left, bottom - top);
//                                ImageUtils.saveBitmap(keydotBitmap);
//
//                                try (FileOutputStream out = new FileOutputStream("/keydots/keydot_" + total + ".png")) {
//                                    keydotBitmap.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
//                                    // PNG is a lossless format, the compression factor (100) is ignored
//                                } catch (IOException e) {
//                                    e.printStackTrace();
//                                }
                                Date currentTime = Calendar.getInstance().getTime();
                                String path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).toString();
                                OutputStream fOut = null;
                                Integer counter = 0;
                                File file = new File(path, "keydot/keydot_" + currentTime.getTime() + ".jpg"); // the File to save , append increasing numeric counter to prevent files from getting overwritten.
                                try {
                                    fOut = new FileOutputStream(file);
//                                Bitmap pictureBitmap = getImageBitmap(myurl); // obtaining the Bitmap
                                    keydotBitmap.compress(Bitmap.CompressFormat.JPEG, 100, fOut); // saving the Bitmap to a file compressed as a JPEG with 85% compression rate
                                    fOut.flush(); // Not really required
                                    fOut.close(); // do not forget to close the stream

                                    MediaStore.Images.Media.insertImage(getContentResolver(),file.getAbsolutePath(),file.getName(),file.getName());
                                } catch (FileNotFoundException e) {
                                    e.printStackTrace();
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }

                                total++;

                                int width = keydotBitmap.getWidth();
                                int height = keydotBitmap.getHeight();

                                int size = keydotBitmap.getRowBytes() * height;
                                ByteBuffer byteBuffer = ByteBuffer.allocate(size);
                                keydotBitmap.copyPixelsToBuffer(byteBuffer);
                                byte[] byteArray = byteBuffer.array();

                                PlanarYUVLuminanceSource planarYUVLuminanceSource = new PlanarYUVLuminanceSource(
                                        byteArray, width, height,
                                        0, 0, width, height,
                                        false
                                );

//                                keydotBitmap.recycle();

//                                var invert = planarYUVLuminanceSource.invert();

                                BinaryBitmap bitmap = new BinaryBitmap(
                                        new HybridBinarizer(
                                                planarYUVLuminanceSource.invert()
                                        )
                                );

                                String resultText = "";

                                try {
                                    Result resultJava = readerJava.decode(bitmap, hints);

                                    if(resultJava != null) {
                                        resultText = resultJava.getBarcodeFormat() + ": " + resultJava.getText();
                                    }

                                }catch (final Exception e) {
                                    e.printStackTrace();
                                    LOGGER.e(e, "Exception detectando keydot!");
                                    Toast toast =
                                            Toast.makeText(
                                                    getApplicationContext(), "Erro ao detectar keydot", Toast.LENGTH_SHORT);
                                    toast.show();
                                }
//
//                                try {
//
//                                    readerCpp.setOptions(new BarcodeReader.Options(
//                                            new HashSet(Arrays.asList(BarcodeReader.Format.DATA_MATRIX)),
//                                            false,
//                                            false
//                                    ));
//
//                                    BarcodeReader.Result reconhecido = readerCpp.read(keydotBitmap, rect, sensorOrientation);
////                      runtime2 += reconhecido?.time?.toInt() ?: 0
//                                    if(reconhecido != null) {
//                                        resultText = reconhecido.getFormat() + ": " + reconhecido.getText();
//
//                                    }
//                                } catch (final Throwable e) {
//                                    e.printStackTrace();
//                                    LOGGER.e(e, "Exception detectando keydot!");
//                                    Toast toast =
//                                            Toast.makeText(
//                                                    getApplicationContext(), "Erro ao detectar keydot", Toast.LENGTH_SHORT);
//                                    toast.show();
////                      finish();
//                                }

                                // DESENHO DO QUADRADO NA IMAGEM
                                canvas.drawRect(location, paint);
                                canvas.drawText(resultText, location.left, location.top + 10, paint);

                                cropToFrameTransform.mapRect(location);

                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }
                        }

                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        trackingOverlay.postInvalidate();

                        computingDetection = false;

                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showFrameInfo(previewWidth + "x" + previewHeight);
                                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });
                    }
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(
                () -> {
                    try {
                        detector.setUseNNAPI(isChecked);
                    } catch (UnsupportedOperationException e) {
                        LOGGER.e(e, "Failed to set \"Use NNAPI\".");
                        runOnUiThread(
                                () -> {
                                    Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                                });
                    }
                });
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }
}
