import torch
import numpy as np
import cv2
import time
from lightglue import SuperPoint, LightGlue

device = torch.device('cuda')
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)

def load_clahe_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = torch.from_numpy(img).float()[None, None] / 255.0
    return img.to(device)

image0 = load_clahe_image('test_frame.png')
image1 = load_clahe_image('reference_map.png')

# Warmup
feats0 = extractor({'image': image0})
feats1 = extractor({'image': image1})
matcher({'image0': feats0, 'image1': feats1})

# Benchmark extraction
ext_times = []
for _ in range(10):
    t0 = time.perf_counter()
    feats0 = extractor({'image': image0})
    ext_times.append((time.perf_counter() - t0) * 1000)

# Benchmark matching
match_times = []
for _ in range(10):
    t0 = time.perf_counter()
    matcher({'image0': feats0, 'image1': feats1})
    match_times.append((time.perf_counter() - t0) * 1000)

print(f"Python Extraction (1 img): {np.mean(ext_times):.2f} ms")
print(f"Python Matching: {np.mean(match_times):.2f} ms")
print(f"Python Total Pipeline (2 img + match): {np.mean(ext_times)*2 + np.mean(match_times):.2f} ms")
