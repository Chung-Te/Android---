import cv2
import numpy as np

# row anchors（跟訓練時一致）
ROW_ANCHORS = list(range(160, 720, 10))[:56]
GRIDING_NUM = 100
NUM_LANES = 4
IMG_W = 800
IMG_H = 288

class LaneTracker:
    def __init__(self):
        self.left_coeffs = None
        self.right_coeffs = None
        self.alpha = 0.7

    def update(self, lanes):
        if len(lanes) >= 2:
            sorted_lanes = sorted(lanes, key=lambda l: np.array(l)[:,0].mean())
            left_pts = np.array(sorted_lanes[0])
            right_pts = np.array(sorted_lanes[-1])
            if len(left_pts) >= 3:
                new_left = np.polyfit(left_pts[:,1], left_pts[:,0], deg=2)
                self.left_coeffs = (self.alpha * self.left_coeffs + (1-self.alpha) * new_left
                                    if self.left_coeffs is not None else new_left)
            if len(right_pts) >= 3:
                new_right = np.polyfit(right_pts[:,1], right_pts[:,0], deg=2)
                self.right_coeffs = (self.alpha * self.right_coeffs + (1-self.alpha) * new_right
                                     if self.right_coeffs is not None else new_right)

    def get_center_offset(self, frame_width, y_ref):
        if self.left_coeffs is None or self.right_coeffs is None:
            return None, "DETECTING"
        left_x = np.polyval(self.left_coeffs, y_ref)
        right_x = np.polyval(self.right_coeffs, y_ref)
        lane_center = (left_x + right_x) / 2
        car_center = frame_width / 2
        offset = car_center - lane_center
        if offset > 50:
            return offset, "RIGHT"
        elif offset < -50:
            return offset, "LEFT"
        else:
            return offset, "NORMAL"

_tracker = LaneTracker()

def draw_lanes(nv21_bytes, width, height, onnx_output_flat):
    """
    Java 呼叫這個函數
    nv21_bytes: 相機影像
    onnx_output_flat: Java ONNX 推論結果（展平的 float list）
    回傳 PNG bytes
    """
    global _tracker

    # NV21 → BGR
    nv21 = np.frombuffer(bytes(nv21_bytes), dtype=np.uint8)
    bgr = cv2.cvtColor(
        nv21.reshape(height * 3 // 2, width),
        cv2.COLOR_YUV2BGR_NV21
    )
    bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    # 更新寬高
    width, height = height, width

    # 解析 ONNX 輸出 [101, 56, 4]
    import struct
    if hasattr(onnx_output_flat, 'toJava'):
        raw = bytes(onnx_output_flat.toJava(bytearray))
        output = np.frombuffer(raw, dtype=np.float32)
    else:
        output = np.array(onnx_output_flat, dtype=np.float32)
    output = output.reshape(101, 56, NUM_LANES)

    col_sample = np.linspace(0, IMG_W - 1, GRIDING_NUM)

    lanes = []
    for lane_idx in range(NUM_LANES):
        lane_pts = []
        for row_idx in range(len(ROW_ANCHORS)):
            prob = output[:GRIDING_NUM, row_idx, lane_idx]
            if prob.max() < 0.5:
                continue
            loc = np.argmax(prob)
            x = int(col_sample[loc] * width / IMG_W)
            y = int(ROW_ANCHORS[row_idx] * height / 720)
            lane_pts.append((x, y))
        if len(lane_pts) > 2:
            lanes.append(lane_pts)

    # 更新追蹤器
    _tracker.update(lanes)
    offset, status = _tracker.get_center_offset(width, int(height * 0.8))

    # 畫車道線
    vis = bgr.copy()
    for i, lane in enumerate(lanes):
        color = (0, 255, 0) if i in [1, 2] else (255, 100, 0)
        pts = np.array(lane)
        if len(pts) >= 3:
            coeffs = np.polyfit(pts[:,1], pts[:,0], deg=2)
            y_range = np.linspace(pts[:,1].min(), pts[:,1].max(), 50)
            x_range = np.polyval(coeffs, y_range)
            for j in range(len(y_range)-1):
                pt1 = (int(x_range[j]), int(y_range[j]))
                pt2 = (int(x_range[j+1]), int(y_range[j+1]))
                if 0 <= pt1[0] < width and 0 <= pt2[0] < width:
                    cv2.line(vis, pt1, pt2, color, 3)

    # 警示
    if status == "LEFT":
        cv2.rectangle(vis, (0, 0), (width-1, height-1), (0, 0, 255), 10)
        cv2.putText(vis, "DEPARTURE LEFT!", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    elif status == "RIGHT":
        cv2.rectangle(vis, (0, 0), (width-1, height-1), (0, 0, 255), 10)
        cv2.putText(vis, "DEPARTURE RIGHT!", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        cv2.putText(vis, status, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    if offset is not None:
        cv2.putText(vis, f"Offset: {int(offset)}px", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    _, buf = cv2.imencode('.png', vis)
    return buf.tobytes()