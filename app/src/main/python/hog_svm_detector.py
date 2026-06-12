import cv2
import numpy as np
import joblib
import os
import glob
from skimage.feature import hog
from skimage.transform import resize

_pipeline = None
_model_path = None

def set_model_path(path: str):
    """從 Java 端傳入正確路徑"""
    global _model_path
    _model_path = path
def _load_model():
    global _pipeline
    if _pipeline is None:
        if _model_path is None:
            raise FileNotFoundError("model path not set, call set_model_path() first")
        import joblib
        _pipeline = joblib.load(_model_path)
    return _pipeline

IMG_SIZE = (64, 64)
CLASS_NAMES = ['Rock', 'Paper', 'Scissors']

def extract_hog_features(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    resized = resized.astype(np.float32) / 255.0
    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )
    return features

def detect_hog_svm(nv21_bytes: bytes, w: int, h: int) -> bytes:
    # ── 1. NV21 → BGR + 旋轉 ──
    yuv = np.frombuffer(bytes(nv21_bytes), dtype=np.uint8).reshape((h * 3 // 2, w))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    h, w = bgr.shape[:2]

    # ── 2. ROI ──
    roi_x, roi_y = int(w*0.20), int(h*0.20)
    roi_w, roi_h = int(w*0.60), int(h*0.60)
    roi = bgr[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    label = "No Hand"
    conf  = 0.0

    try:
        # ── 3. 膚色偵測確認有手 ──
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([0, 20, 70]),
                           np.array([20, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_cnt = max(contours, key=cv2.contourArea)
            area    = cv2.contourArea(max_cnt)

            if area > 3000:
                # ── 4. HoG 特徵 + SVM 推論 ──
                features = extract_hog_features(roi)
                pipeline = _load_model()
                pred_id  = pipeline.predict([features])[0]
                proba    = pipeline.predict_proba([features])[0]
                conf     = proba[pred_id]
                label    = CLASS_NAMES[pred_id]

                cv2.drawContours(roi, [max_cnt], -1, (0, 255, 0), 2)

    except Exception as e:
        label = f"Err:{str(e)[:20]}"

    # ── 5. 畫框與文字 ──
    cv2.rectangle(bgr, (roi_x, roi_y),
                  (roi_x+roi_w, roi_y+roi_h), (0, 255, 255), 3)

    color_map = {
        'Rock': (0,0,255), 'Paper': (0,255,0),
        'Scissors': (0,255,255), 'No Hand': (128,128,128)
    }
    color = color_map.get(label, (255,255,255))

    cv2.putText(bgr, label,
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
    cv2.putText(bgr, f"conf:{conf:.2f}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("encode failed")
    return buf.tobytes()