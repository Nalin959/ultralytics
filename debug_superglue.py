import cv2
import numpy as np
import os
import sys
import torch

BASE = "/home/nalin/gps_denied_project"
sys.path.append(os.path.join(BASE, "SuperGluePretrainedNetwork"))
sys.path.append(BASE)

from models.matching import Matching
from models.utils import frame2tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 2048
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}
matcher = Matching(config).eval().to(device)

test_gray_path = os.path.join(BASE, "scratch_test_gray.png")
ref_gray_path = os.path.join(BASE, "scratch_ref_gray.png")

img0 = cv2.imread(ref_gray_path, cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread(test_gray_path, cv2.IMREAD_GRAYSCALE)

print(f"Original img0 shape: {img0.shape}")
print(f"Original img1 shape: {img1.shape}")

max_dim = 1024
scale0 = 1.0
scale1 = 1.0

if max(img0.shape) > max_dim:
    scale0 = max_dim / max(img0.shape)
    img0 = cv2.resize(img0, (int(img0.shape[1]*scale0), int(img0.shape[0]*scale0)))
if max(img1.shape) > max_dim:
    scale1 = max_dim / max(img1.shape)
    img1 = cv2.resize(img1, (int(img1.shape[1]*scale1), int(img1.shape[0]*scale1)))

print(f"Resized img0 shape: {img0.shape}")
print(f"Resized img1 shape: {img1.shape}")
print(f"Scale0: {scale0}, Scale1: {scale1}")

tensor0 = frame2tensor(img0, device)
tensor1 = frame2tensor(img1, device)

with torch.inference_mode():
    pred = matcher({'image0': tensor0, 'image1': tensor1})
    
kpts0 = pred['keypoints0'][0].cpu().numpy()
kpts1 = pred['keypoints1'][0].cpu().numpy()
matches = pred['matches0'][0].cpu().numpy()
confidence = pred['matching_scores0'][0].cpu().numpy()

valid = matches > -1
pts0 = kpts0[valid]
pts1 = kpts1[matches[valid]]
mconf = confidence[valid]

print(f"Found {len(pts0)} matches")
print(f"First 5 raw pts0 (1024 space):\n{pts0[:5]}")
print(f"First 5 raw pts1 (1024 space):\n{pts1[:5]}")

pts0 = pts0 / scale0
pts1 = pts1 / scale1

print(f"First 5 restored pts0 (original space):\n{pts0[:5]}")
print(f"First 5 restored pts1 (original space):\n{pts1[:5]}")

img0_cv = cv2.imread(ref_gray_path)
h0, w0 = img0_cv.shape[:2]
s0 = 720 / h0
print(f"Visualization s0 scale: {s0}")
print(f"First 5 final vis pts0 (720 space):\n{pts0[:5] * s0}")
