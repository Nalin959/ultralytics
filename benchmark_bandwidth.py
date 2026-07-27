import cv2
import numpy as np
import os
import sys
import torch
import time

BASE = "/home/nalin/gps_denied_project"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get image paths
ref_path = os.path.join(BASE, "reference_map.png")
test_path = os.path.join(BASE, "test_frame.png")

print(f"Benchmarking on {device.upper()}")
print("Image sizes:")
img0 = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
print(f" - Reference: {img0.shape}")
print(f" - Test Frame: {img1.shape}")
print("="*60)

results = []

def print_stats(name, latency, vram, notes):
    print(f"\n[{name}]")
    print(f"  Latency:    {latency:.2f} ms")
    print(f"  VRAM Usage: {vram:.2f} MB")
    print(f"  Notes:      {notes}")
    results.append((name, latency, vram))

# =========================================================
# 1. LIGHTGLUE
# =========================================================
print("\nLoading LightGlue...")
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

extractor = SuperPoint(max_num_keypoints=2048, keypoint_threshold=0.04).eval().to(device)
matcher_lg = LightGlue(features='superpoint').eval().to(device)

image0 = load_image(ref_path).to(device)
image1 = load_image(test_path).to(device)

# Warmup
with torch.inference_mode():
    f0 = extractor.extract(image0)
    f1 = extractor.extract(image1)
    _ = matcher_lg({'image0': f0, 'image1': f1})

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

start = time.time()
with torch.inference_mode():
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher_lg({'image0': feats0, 'image1': feats1})
torch.cuda.synchronize()
end = time.time()

vram_lg = torch.cuda.max_memory_allocated() / (1024**2)
lat_lg = (end - start) * 1000
print_stats("LightGlue (SuperPoint+LightGlue)", lat_lg, vram_lg, "Adaptive sparse matching, no resizing needed")

del extractor, matcher_lg, feats0, feats1, matches01, image0, image1
torch.cuda.empty_cache()

# =========================================================
# 2. SUPERGLUE
# =========================================================
print("\nLoading SuperGlue...")
sys.path.append(os.path.join(BASE, "SuperGluePretrainedNetwork"))
from models.matching import Matching
from models.utils import frame2tensor

config_sg = {
    'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 2048},
    'superglue': {'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}
}
matcher_sg = Matching(config_sg).eval().to(device)

max_dim = 1024
scale0 = max_dim / max(img0.shape) if max(img0.shape) > max_dim else 1.0
scale1 = max_dim / max(img1.shape) if max(img1.shape) > max_dim else 1.0
img0_sg = cv2.resize(img0, (int(img0.shape[1]*scale0), int(img0.shape[0]*scale0)))
img1_sg = cv2.resize(img1, (int(img1.shape[1]*scale1), int(img1.shape[0]*scale1)))

tensor0 = frame2tensor(img0_sg, device)
tensor1 = frame2tensor(img1_sg, device)

# Warmup
with torch.inference_mode():
    _ = matcher_sg({'image0': tensor0, 'image1': tensor1})

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

start = time.time()
with torch.inference_mode():
    pred = matcher_sg({'image0': tensor0, 'image1': tensor1})
torch.cuda.synchronize()
end = time.time()

vram_sg = torch.cuda.max_memory_allocated() / (1024**2)
lat_sg = (end - start) * 1000
print_stats("SuperGlue (SuperPoint+SuperGlue)", lat_sg, vram_sg, "Fixed sparse matching, required downscaling to 1024 to prevent OOM")

del matcher_sg, pred, tensor0, tensor1
torch.cuda.empty_cache()

# =========================================================
# 3. LoFTR
# =========================================================
print("\nLoading EfficientLoFTR...")
from transformers import AutoImageProcessor, AutoModel
processor = AutoImageProcessor.from_pretrained("zju-community/efficientloftr")
model = AutoModel.from_pretrained("zju-community/efficientloftr", trust_remote_code=True)
try:
    model = model.eval().to(device)
except Exception as e:
    print(f"Warning: could not move LoFTR to GPU, running on CPU. {e}")
    device = 'cpu'

# We MUST resize for LoFTR, otherwise it OOMs completely on RTX 4050
img0_loftr = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
img1_loftr = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
img0_loftr = cv2.resize(img0_loftr, (640, 480))
img1_loftr = cv2.resize(img1_loftr, (640, 480))

inputs = processor(images=[img0_loftr, img1_loftr], return_tensors="pt")
if device == 'cuda':
    inputs = {k: v.to(device) for k, v in inputs.items()}

# Warmup
with torch.inference_mode():
    _ = model(**inputs)

if device == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

start = time.time()
with torch.inference_mode():
    outputs = model(**inputs)
if device == 'cuda':
    torch.cuda.synchronize()
end = time.time()

vram_loftr = torch.cuda.max_memory_allocated() / (1024**2) if device == 'cuda' else 0.0
lat_loftr = (end - start) * 1000
print_stats("EfficientLoFTR", lat_loftr, vram_loftr, f"Dense matching, aggressively downscaled to 640x480. Device: {device}")

print("\n" + "="*60)
print("BENCHMARK SUMMARY")
print("="*60)
for r in results:
    print(f"{r[0]:<35} | {r[1]:>8.2f} ms | {r[2]:>8.2f} MB VRAM")
