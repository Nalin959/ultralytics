import cv2
import numpy as np
import os
import sys
import torch
import matplotlib.cm as cm

BASE = "/home/nalin/gps_denied_project"
sys.path.append(os.path.join(BASE, "SuperGluePretrainedNetwork"))
sys.path.append(BASE)

from models.matching import Matching
from models.utils import frame2tensor

# Initialize SuperGlue
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

def run_superglue_sanity(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to prevent OOM
    max_dim = 1024
    if max(img.shape) > max_dim:
        scale = max_dim / max(img.shape)
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
        
    tensor = frame2tensor(img, device)
    
    torch.cuda.empty_cache()
    with torch.inference_mode():
        # Matching the exact same image to itself
        pred = matcher({'image0': tensor, 'image1': tensor})
        
    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()
    
    valid = matches > -1
    pts0 = kpts0[valid]
    pts1 = kpts1[matches[valid]]
    mconf_np = confidence[valid]
    
    n_matches = len(pts0)
    avg_conf = np.mean(mconf_np) if n_matches > 0 else 0.0

    print(f"Sanity Check: Self-Matching")
    print(f"Total Matches: {n_matches}")
    print(f"Mean Confidence: {avg_conf:.4f}")

    # Visualization
    img_cv = cv2.imread(img_path)
    h, w = img_cv.shape[:2]
    s = 720 / h
    img_res = cv2.resize(img_cv, (int(w * s), 720))
    pts0_res = pts0 * s
    pts1_res = pts1 * s
        
    combined = np.hstack([img_res, img_res])
    offset = img_res.shape[1]
    
    color = cm.jet(mconf_np)
    
    for i in range(len(pts0_res)):
        p0 = (int(pts0_res[i][0]), int(pts0_res[i][1]))
        p1 = (int(pts1_res[i][0]) + offset, int(pts1_res[i][1]))
        c = (int(color[i][2]*255), int(color[i][1]*255), int(color[i][0]*255))
        cv2.line(combined, p0, p1, c, 1)
        cv2.circle(combined, p0, 2, c, -1)
        cv2.circle(combined, p1, 2, c, -1)
        
    text = f"Matches: {n_matches} | Mean Conf: {avg_conf:.4f}"
    cv2.putText(combined, text, (10, 720 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    out_path = os.path.join(BASE, "comparison_superglue_sanity.jpg")
    cv2.imwrite(out_path, combined)
    print(f"Saved sanity check visualization to {out_path}")

if __name__ == "__main__":
    ref_event = os.path.join(BASE, "reference_map.png")
    run_superglue_sanity(ref_event)
