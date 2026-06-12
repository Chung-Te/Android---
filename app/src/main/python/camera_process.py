import numpy as np
import cv2

def process_nv21(nv21_bytes: bytes, w: int, h: int) -> bytes:
    yuv = np.frombuffer(bytes(nv21_bytes), dtype=np.uint8).reshape((h * 3 // 2, w))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 80, 160)
    out = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("encode failed")
    return buf.tobytes()