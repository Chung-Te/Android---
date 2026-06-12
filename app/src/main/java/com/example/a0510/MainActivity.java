package com.example.a0510;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.camera.core.*;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";

    // ── 第 8 章 Views ──
    private Button btnPickImage, btnProcessImage;
    private TextView txtStatus;
    private ImageView imgOriginal, imgResult;
    private byte[] currentImageBytes;

    // ── 相機 Views ──
    private PreviewView previewView;
    private ImageView imgCameraResult;
    private TextView txtCameraStatus;
    private Button btnCapture;
    private Button btnToggleMode;

    // ── CameraX ──
    private ImageCapture imageCapture;
    private ExecutorService cameraExecutor;
    private Python py;

    // ── ImageAnalysis 節流 ──
    private final AtomicBoolean isProcessing = new AtomicBoolean(false);
    private int frameCount = 0;
    private static final int PROCESS_EVERY_N_FRAMES = 5;

    // ── 模式切換 ──
    private boolean rpsMode    = false;
    private boolean hogSvmMode = false;

    // ── 圖片選取器 ──
    private ActivityResultLauncher<String> pickImageLauncher;

    // ── 相機權限請求 ──
    private final ActivityResultLauncher<String> cameraPermLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(),
                    isGranted -> {
                        if (isGranted) startCamera();
                        else Toast.makeText(this, "需要攝影機權限", Toast.LENGTH_LONG).show();
                    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 初始化 Chaquopy
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
        py = Python.getInstance();

        // 複製模型到 files 目錄
        copyAssetToFiles("rps_hog_svm.pkl");

        // 複製完模型後，把路徑傳給 Python
        File modelFile = new File(getFilesDir(), "rps_hog_svm.pkl");
        copyAssetToFiles("rps_hog_svm.pkl");

        // 告訴 Python 模型在哪
        PyObject detector = py.getModule("hog_svm_detector");
        detector.callAttr("set_model_path", modelFile.getAbsolutePath());
        Log.d(TAG, "Model path: " + modelFile.getAbsolutePath());

        // ── 第 8 章 View 綁定 ──
        btnPickImage    = findViewById(R.id.btnPickImage);
        btnProcessImage = findViewById(R.id.btnProcessImage);
        txtStatus       = findViewById(R.id.txtStatus);
        imgOriginal     = findViewById(R.id.imgOriginal);
        imgResult       = findViewById(R.id.imgResult);

        // ── 相機 View 綁定 ──
        previewView     = findViewById(R.id.previewView);
        imgCameraResult = findViewById(R.id.imgCameraResult);
        txtCameraStatus = findViewById(R.id.txtCameraStatus);
        btnCapture      = findViewById(R.id.btnCapture);
        btnToggleMode   = findViewById(R.id.btnToggleMode);
        btnCapture.setEnabled(false);

        cameraExecutor = Executors.newSingleThreadExecutor();

        // ── 第 8 章按鈕 ──
        pickImageLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> { if (uri != null) loadImageFromUri(uri); }
        );
        btnPickImage.setOnClickListener(v -> pickImageLauncher.launch("image/*"));
        btnProcessImage.setOnClickListener(v -> processImageStatic());

        // ── 擷取按鈕 ──
        btnCapture.setOnClickListener(v -> captureAndProcess());

        // ── 模式切換按鈕 ──
        btnToggleMode.setOnClickListener(v -> {
            if (!rpsMode && !hogSvmMode) {
                rpsMode = true; hogSvmMode = false;
                btnToggleMode.setText("Switch to HoG+SVM");
                txtCameraStatus.setText("RPS Mode");
            } else if (rpsMode) {
                rpsMode = false; hogSvmMode = true;
                btnToggleMode.setText("Switch to Canny");
                txtCameraStatus.setText("HoG+SVM Mode");
            } else {
                rpsMode = false; hogSvmMode = false;
                btnToggleMode.setText("Switch to RPS");
                txtCameraStatus.setText("Canny Mode");
            }
        });

        // ── 啟動相機 ──
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            cameraPermLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    // ════════════════════════════════════════
    // 第 8 章：靜態圖片
    // ════════════════════════════════════════

    private void loadImageFromUri(Uri uri) {
        try {
            InputStream is = getContentResolver().openInputStream(uri);
            currentImageBytes = readAllBytes(is);
            Bitmap bmp = BitmapFactory.decodeByteArray(
                    currentImageBytes, 0, currentImageBytes.length);
            imgOriginal.setImageBitmap(bmp);
            txtStatus.setText("Loaded from device");
        } catch (Exception e) {
            txtStatus.setText("Load failed: " + e.getMessage());
        }
    }

    private void processImageStatic() {
        if (currentImageBytes == null) {
            txtStatus.setText("No image loaded");
            return;
        }
        txtStatus.setText("Processing...");
        byte[] snapshot = currentImageBytes;
        new Thread(() -> {
            try {
                PyObject module = py.getModule("opencv_process");
                PyObject result = module.callAttr(
                        "canny_from_image_bytes", (Object) snapshot);
                byte[] outPng = result.toJava(byte[].class);
                Bitmap bmp = BitmapFactory.decodeByteArray(outPng, 0, outPng.length);
                runOnUiThread(() -> {
                    imgResult.setImageBitmap(bmp);
                    txtStatus.setText("Done");
                });
            } catch (Exception e) {
                runOnUiThread(() -> txtStatus.setText("Error: " + e.getMessage()));
            }
        }).start();
    }

    // ════════════════════════════════════════
    // CameraX
    // ════════════════════════════════════════

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future =
                ProcessCameraProvider.getInstance(this);

        future.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = future.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                imageCapture = new ImageCapture.Builder()
                        .setTargetResolution(new Size(640, 480))
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build();

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor, imageProxy -> {
                    frameCount++;
                    if (frameCount % PROCESS_EVERY_N_FRAMES != 0
                            || !isProcessing.compareAndSet(false, true)) {
                        imageProxy.close();
                        return;
                    }
                    try {
                        byte[] nv21  = imageProxyToNv21(imageProxy);
                        int   width  = imageProxy.getWidth();
                        int   height = imageProxy.getHeight();
                        imageProxy.close();

                        PyObject module, result;
                        if (hogSvmMode) {
                            module = py.getModule("hog_svm_detector");
                            result = module.callAttr("detect_hog_svm", nv21, width, height);
                        } else if (rpsMode) {
                            module = py.getModule("rps_detector");
                            result = module.callAttr("detect_rps", nv21, width, height);
                        } else {
                            module = py.getModule("camera_process");
                            result = module.callAttr("process_nv21", nv21, width, height);
                        }

                        byte[] outPng = result.toJava(byte[].class);
                        Bitmap bmp = BitmapFactory.decodeByteArray(outPng, 0, outPng.length);
                        String modeStr = hogSvmMode ? "HoG+SVM" : (rpsMode ? "RPS" : "Canny");
                        runOnUiThread(() -> {
                            imgCameraResult.setImageBitmap(bmp);
                            txtCameraStatus.setText(modeStr + " ✓ frame=" + frameCount);
                        });
                    } catch (Exception e) {
                        Log.e(TAG, "Analysis error", e);
                        runOnUiThread(() ->
                                txtCameraStatus.setText("Error: " + e.getMessage()));
                    } finally {
                        isProcessing.set(false);
                    }
                });

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(
                        this,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview, imageCapture, imageAnalysis
                );

                runOnUiThread(() -> {
                    btnCapture.setEnabled(true);
                    txtCameraStatus.setText("Camera ready");
                });

            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "startCamera failed", e);
                runOnUiThread(() -> txtCameraStatus.setText("Camera init failed"));
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void captureAndProcess() {
        if (imageCapture == null) return;
        btnCapture.setEnabled(false);
        txtCameraStatus.setText("Capturing...");

        imageCapture.takePicture(cameraExecutor,
                new ImageCapture.OnImageCapturedCallback() {
                    @Override
                    public void onCaptureSuccess(@NonNull ImageProxy imageProxy) {
                        byte[] nv21  = imageProxyToNv21(imageProxy);
                        int   width  = imageProxy.getWidth();
                        int   height = imageProxy.getHeight();
                        imageProxy.close();
                        try {
                            PyObject module = py.getModule("camera_process");
                            PyObject result = module.callAttr(
                                    "process_nv21", nv21, width, height);
                            byte[] outPng = result.toJava(byte[].class);
                            Bitmap bmp = BitmapFactory.decodeByteArray(outPng, 0, outPng.length);
                            runOnUiThread(() -> {
                                imgCameraResult.setImageBitmap(bmp);
                                txtCameraStatus.setText("Captured ✓");
                                btnCapture.setEnabled(true);
                            });
                        } catch (Exception e) {
                            Log.e(TAG, "Python error", e);
                            runOnUiThread(() -> {
                                txtCameraStatus.setText("Error: " + e.getMessage());
                                btnCapture.setEnabled(true);
                            });
                        }
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException e) {
                        Log.e(TAG, "Capture error", e);
                        runOnUiThread(() -> {
                            txtCameraStatus.setText("Capture failed");
                            btnCapture.setEnabled(true);
                        });
                    }
                });
    }

    // ════════════════════════════════════════
    // 第 11 章：ImageProxy → NV21
    // ════════════════════════════════════════

    private static byte[] imageProxyToNv21(ImageProxy image) {
        int w = image.getWidth();
        int h = image.getHeight();
        byte[] nv21 = new byte[w * h * 3 / 2];

        ImageProxy.PlaneProxy yPlane = image.getPlanes()[0];
        ByteBuffer yBuf = yPlane.getBuffer();
        int yRowStride = yPlane.getRowStride();
        int pos = 0;
        for (int r = 0; r < h; r++) {
            yBuf.position(r * yRowStride);
            yBuf.get(nv21, pos, w);
            pos += w;
        }

        ImageProxy.PlaneProxy uPlane = image.getPlanes()[1];
        ImageProxy.PlaneProxy vPlane = image.getPlanes()[2];
        ByteBuffer uBuf = uPlane.getBuffer();
        ByteBuffer vBuf = vPlane.getBuffer();
        int rowStrideUV   = uPlane.getRowStride();
        int pixelStrideUV = uPlane.getPixelStride();

        for (int r = 0; r < h / 2; r++) {
            int rowStart = r * rowStrideUV;
            for (int c = 0; c < w / 2; c++) {
                int idx = rowStart + c * pixelStrideUV;
                nv21[pos++] = vBuf.get(idx);
                nv21[pos++] = uBuf.get(idx);
            }
        }
        return nv21;
    }

    // ════════════════════════════════════════
    // 共用工具
    // ════════════════════════════════════════

    private void copyAssetToFiles(String filename) {
        File outFile = new File(getFilesDir(), filename);
        if (outFile.exists()) return;
        try {
            InputStream in   = getAssets().open(filename);
            OutputStream out = new java.io.FileOutputStream(outFile);
            byte[] buf = new byte[4096];
            int len;
            while ((len = in.read(buf)) != -1) out.write(buf, 0, len);
            in.close();
            out.close();
            Log.d(TAG, "Copied " + filename + " to " + outFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e(TAG, "copyAssetToFiles failed: " + e.getMessage());
        }
    }

    private byte[] readAllBytes(InputStream is) throws Exception {
        ByteArrayOutputStream buf = new ByteArrayOutputStream();
        byte[] chunk = new byte[4096];
        int n;
        while ((n = is.read(chunk)) != -1) buf.write(chunk, 0, n);
        is.close();
        return buf.toByteArray();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) cameraExecutor.shutdown();
    }
}