import numpy as np
def array_sum(values):
    arr = np.array(values)
    return int(arr.sum())

def normalize_byte_values(data):
    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    arr = arr.astype(np.float32) / 255.0
    return arr.tolist()

def make_gradient(width, height):
    x = np.linspace(0, 255, width, dtype=np.uint8)
    img = np.tile(x, (height, 1))
    return img.tobytes()