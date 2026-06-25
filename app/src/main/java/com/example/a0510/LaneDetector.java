package com.example.a0510;

import ai.onnxruntime.*;
import android.content.Context;
import android.util.Log;
import java.io.*;
import java.nio.FloatBuffer;
import java.util.*;

public class LaneDetector {
    private static final String TAG = "LaneDetector";
    private OrtEnvironment env;
    private OrtSession session;
    private static final int INPUT_W = 800;
    private static final int INPUT_H = 288;
    private static final int GRIDING_NUM = 100;
    private static final int NUM_LANES = 4;
    private static final int ROW_ANCHORS = 56;

    public LaneDetector(Context context) throws Exception {
        env = OrtEnvironment.getEnvironment();
        // 從 files 目錄載入模型
        File modelFile = new File(context.getFilesDir(), "ufld_lane.onnx");
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        session = env.createSession(modelFile.getAbsolutePath(), opts);
        Log.d(TAG, "ONNX model loaded");
    }

    public float[] run(float[] inputData) throws Exception {
        long[] shape = {1, 3, INPUT_H, INPUT_W};
        OnnxTensor tensor = OnnxTensor.createTensor(env,
                FloatBuffer.wrap(inputData), shape);
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input", tensor);
        OrtSession.Result result = session.run(inputs);
        float[][][][] output = (float[][][][]) result.get(0).getValue();
        // output shape: [1][101][56][4]
        // 展平成 1D
        float[] flat = new float[101 * ROW_ANCHORS * NUM_LANES];
        int idx = 0;
        for (int i = 0; i < 101; i++)
            for (int j = 0; j < ROW_ANCHORS; j++)
                for (int k = 0; k < NUM_LANES; k++)
                    flat[idx++] = output[0][i][j][k];
        tensor.close();
        result.close();
        return flat;
    }

    public void close() throws Exception {
        session.close();
        env.close();
    }
}