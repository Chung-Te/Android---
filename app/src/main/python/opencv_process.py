import cv2
import numpy as np

def canny_from_image_bytes(image_bytes):
    print("Python OpenCV processing started")
    data = np.frombuffer(bytes(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode failed")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 80, 160)
    ok, buf = cv2.imencode(".png", edge)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    print("Python OpenCV processing finished")
    return buf.tobytes()