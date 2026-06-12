import cv2
import numpy as np

def detect_rps(nv21_bytes: bytes, w: int, h: int) -> bytes:
    # ── 1. NV21 → BGR + 旋轉修正 ──
    yuv = np.frombuffer(bytes(nv21_bytes), dtype=np.uint8).reshape((h * 3 // 2, w))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    h, w = bgr.shape[:2]

    # ── 2. ROI（中央60%）──
    roi_x = int(w * 0.20)
    roi_y = int(h * 0.20)
    roi_w = int(w * 0.60)
    roi_h = int(h * 0.60)
    roi = bgr[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # ── 3. 膚色偵測 ──
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0,  20,  70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # ── 4. 找最大輪廓 ──
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label       = "No Hand"
    area_ratio  = 0.0
    aspect_ratio = 0.0

    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(max_cnt)

        if cnt_area > 3000:
            # ── 5. 凸包面積 ──
            hull_pts  = cv2.convexHull(max_cnt)
            hull_area = cv2.contourArea(hull_pts)

            # 面積比：輪廓面積 / 凸包面積
            area_ratio = cnt_area / hull_area if hull_area > 0 else 0.0

            # ── 6. 外接矩形寬高比 ──
            x, y, rw, rh = cv2.boundingRect(max_cnt)
            # 長邊 / 短邊，永遠 >= 1
            aspect_ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 1.0

            # ── 7. 判斷邏輯 ──
            # 石頭：面積比高（圓）+ 寬高比接近1（正方形）
            # 布：  面積比低（展開手指，凸包填不滿）+ 寬高比適中
            # 剪刀：面積比中等 + 寬高比偏大（細長）
            if area_ratio >= 0.85:
                label = "Rock"
            elif area_ratio <= 0.75 and aspect_ratio >= 1.4:
                label = "Scissors"
            elif area_ratio <= 0.80:
                label = "Paper"
            else:
                label = "Unknown"

            # 畫輪廓
            cv2.drawContours(roi, [max_cnt], -1, (0, 255, 0), 2)
            cv2.drawContours(roi, [hull_pts], -1, (0, 0, 255), 2)

            # 畫外接矩形
            cv2.rectangle(roi, (x, y), (x+rw, y+rh), (255, 165, 0), 2)

    # ── 8. ROI 框（黃色）──
    cv2.rectangle(bgr,
                  (roi_x, roi_y),
                  (roi_x+roi_w, roi_y+roi_h),
                  (0, 255, 255), 3)

    # ── 9. 結果文字 ──
    color_map = {
        "Rock":     (0, 0, 255),
        "Scissors": (0, 255, 255),
        "Paper":    (0, 255, 0),
        "Unknown":  (200, 200, 0),
        "No Hand":  (128, 128, 128)
    }
    color = color_map.get(label, (255, 255, 255))

    # 大標題
    cv2.putText(bgr, label,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)

    # Debug 數值（調參數用）
    cv2.putText(bgr,
                f"area_ratio={area_ratio:.2f}  aspect={aspect_ratio:.2f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── 10. 輸出 PNG ──
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("encode failed")
    return buf.tobytes()