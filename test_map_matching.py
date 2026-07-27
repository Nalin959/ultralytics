import os
import cv2
import sys
import numpy as np
import gc
import torch

BASE = "/home/nalin/gps_denied_project"
sys.path.append(BASE)

# Import model wrappers from previous scripts
from test_lightglue_confidence import run_lightglue
from compare_sam3_vs_loftr import run_loftr

# Paths to the specific artifacts requested
test_frame_path = "/home/nalin/gps_denied_project/test_frame.png"
ref_map_path = "/home/nalin/gps_denied_project/reference_map.png"

print("==================================================================")
print("  Comparing Drone Frame and Ref Map")
print("==================================================================")
print(f"Ref Map: {ref_map_path}")
print(f"Drone Frame: {test_frame_path}")

if not os.path.exists(test_frame_path) or not os.path.exists(ref_map_path):
    print("Error: Missing input files.")
    sys.exit(1)

# Run LightGlue
print("\n--- Running SuperPoint + LightGlue ---")
lg_vis, lg_n, lg_conf, lg_confs = run_lightglue(ref_map_path, test_frame_path)
lg_out = "/home/nalin/.gemini/antigravity-ide/brain/b9e5284b-9b2a-4983-958c-ed55daa6eb9b/comparison_lightglue_map_vs_drone.jpg"
cv2.imwrite(lg_out, lg_vis)
print(f"LightGlue - Matches: {lg_n}, Mean Confidence: {lg_conf:.4f}")
print(f"Saved visualization to {lg_out}")

# Free memory
gc.collect()
torch.cuda.empty_cache()

# Run LoFTR
print("\n--- Running EfficientLoFTR ---")
loftr_vis, loftr_n, loftr_conf, loftr_confs = run_loftr(ref_map_path, test_frame_path)
loftr_out = "/home/nalin/.gemini/antigravity-ide/brain/b9e5284b-9b2a-4983-958c-ed55daa6eb9b/comparison_loftr_map_vs_drone.jpg"
cv2.imwrite(loftr_out, loftr_vis)
print(f"LoFTR - Matches: {loftr_n}, Mean Confidence: {loftr_conf:.4f}")
print(f"Saved visualization to {loftr_out}")

print("\nDONE!")
