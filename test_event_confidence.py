import cv2
import numpy as np
import os
import gc
import torch
from compare_sam3_vs_loftr import run_sam3, run_loftr

BASE = "/home/nalin/gps_denied_project"
ref_event = os.path.join(BASE, "reference_map.png")
test_event = os.path.join(BASE, "test_frame_event.png")

print("\n--- Running confidence score comparison (RGB Map vs Event Frame) ---")

# SAM3
print("\n  MODEL 1: SAM3 on RGB Map vs Event Frame")
ref_overlay, ref_stats, ref_conf = run_sam3(ref_event, "RGB Ref Map")
test_overlay, test_stats, test_conf = run_sam3(test_event, "Event Test Frame")

TARGET_H = 720
h_r, w_r = ref_overlay.shape[:2]
s_r = TARGET_H / h_r
ref_resized = cv2.resize(ref_overlay, (int(w_r * s_r), TARGET_H))

h_t, w_t = test_overlay.shape[:2]
s_t = TARGET_H / h_t
test_resized = cv2.resize(test_overlay, (int(w_t * s_t), TARGET_H))

sam3_comparison = np.hstack([ref_resized, test_resized])
sam3_out = os.path.join(BASE, "comparison_sam3_event_optB.jpg")
cv2.imwrite(sam3_out, sam3_comparison)
sam3_overall = (ref_conf + test_conf) / 2 if (ref_conf + test_conf) > 0 else 0.0

del ref_overlay, test_overlay
gc.collect()
torch.cuda.empty_cache()

# LoFTR - RAW
print("\n  MODEL 2A: EfficientLoFTR on RAW (RGB Map vs Event Frame)")
loftr_vis_raw, n_matches_raw, loftr_conf_raw, loftr_confs_raw = run_loftr(ref_event, test_event)
loftr_out_raw = os.path.join(BASE, "comparison_loftr_event_raw.jpg")
cv2.imwrite(loftr_out_raw, loftr_vis_raw)

# LoFTR with Spatial Gradient Modality Bridging
print("\n  MODEL 2B: EfficientLoFTR on Spatial Gradient Maps")

# 1. Gradient Map for Reference Map
ref_img = cv2.imread(ref_event)
ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float64)
grad_x = cv2.Scharr(ref_gray, cv2.CV_64F, 1, 0)
grad_y = cv2.Scharr(ref_gray, cv2.CV_64F, 0, 1)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
kernel = np.ones((3,3), np.uint8)
grad_mag = cv2.dilate(grad_mag, kernel, iterations=1)
ref_grad_path = os.path.join(BASE, "ref_map_grad.png")
cv2.imwrite(ref_grad_path, cv2.cvtColor(grad_mag, cv2.COLOR_GRAY2BGR))

# 2. Grayscale Event Frame
event_img = cv2.imread(test_event)
event_gray = cv2.cvtColor(event_img, cv2.COLOR_BGR2GRAY)
event_gray = cv2.dilate(event_gray, kernel, iterations=1)
event_grad_path = os.path.join(BASE, "test_frame_event_grad.png")
cv2.imwrite(event_grad_path, cv2.cvtColor(event_gray, cv2.COLOR_GRAY2BGR))

loftr_vis_grad, n_matches_grad, loftr_conf_grad, loftr_confs_grad = run_loftr(ref_grad_path, event_grad_path)
loftr_out_grad = os.path.join(BASE, "comparison_loftr_event_grad.jpg")
cv2.imwrite(loftr_out_grad, loftr_vis_grad)

# 3. Spatial Gradient Reference Map vs RAW Event Frame
print("\n  MODEL 2C: EfficientLoFTR on Spatial Gradient Map vs RAW Event Frame")
loftr_vis_grad_raw, n_matches_grad_raw, loftr_conf_grad_raw, loftr_confs_grad_raw = run_loftr(ref_grad_path, test_event)
loftr_out_grad_raw = os.path.join(BASE, "comparison_loftr_event_grad_raw.jpg")
cv2.imwrite(loftr_out_grad_raw, loftr_vis_grad_raw)

# Print results
print("\n" + "=" * 70)
print("  RGB MAP VS EVENT FRAME CONFIDENCE SCORE RESULTS")
print("=" * 70)
print(f"  {'Model':<45} {'Confidence':>12}")
print(f"  {'-'*45} {'-'*12}")
print(f"  {'SAM3 (segmentation)':<45} {sam3_overall:>12.4f}")
print(f"  {'EfficientLoFTR (RAW)':<45} {loftr_conf_raw:>12.4f}")
print(f"  {'EfficientLoFTR (Gradient Bridged Grayscale)':<45} {loftr_conf_grad:>12.4f}")
print(f"  {'EfficientLoFTR (Gradient vs RAW Event)':<45} {loftr_conf_grad_raw:>12.4f}")

