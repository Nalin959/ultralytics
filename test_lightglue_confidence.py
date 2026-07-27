import cv2
import numpy as np
import os
import sys
import gc
import torch
import matplotlib.cm as cm

BASE = "/home/nalin/gps_denied_project"
sys.path.append(os.path.join(BASE, "LightGlue"))
sys.path.append(BASE)

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d

# Initialize LightGlue
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features='superpoint').eval().to(device)

def run_lightglue(img0_path, img1_path, top_n=10000):
    image0 = load_image(img0_path).to(device)
    image1 = load_image(img1_path).to(device)
    
    with torch.inference_mode():
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01['matches']
        points0 = feats0['keypoints'][matches[..., 0]]
        points1 = feats1['keypoints'][matches[..., 1]]
        mconf = matches01['scores']

    pts0 = points0.cpu().numpy()
    pts1 = points1.cpu().numpy()
    mconf_np = mconf.cpu().numpy()
    n_matches = len(pts0)

    # Visualization
    img0_cv = cv2.imread(img0_path)
    img1_cv = cv2.imread(img1_path)
    
    # Resize to common height for hstack
    TARGET_H = 720
    h0, w0 = img0_cv.shape[:2]
    h1, w1 = img1_cv.shape[:2]
    s0 = TARGET_H / h0
    s1 = TARGET_H / h1
    
    img0_res = cv2.resize(img0_cv, (int(w0 * s0), TARGET_H))
    img1_res = cv2.resize(img1_cv, (int(w1 * s1), TARGET_H))
    
    pts0_res = pts0 * s0
    pts1_res = pts1 * s1
    
    if n_matches > top_n:
        idx = np.argsort(mconf_np)[::-1][:top_n]
        pts0_vis = pts0_res[idx]
        pts1_vis = pts1_res[idx]
        mconf_vis = mconf_np[idx]
    else:
        pts0_vis = pts0_res
        pts1_vis = pts1_res
        mconf_vis = mconf_np
        
    combined = np.hstack([img0_res, img1_res])
    offset = img0_res.shape[1]
    
    color = cm.jet(mconf_vis)
    
    for i in range(len(pts0_vis)):
        p0 = (int(pts0_vis[i][0]), int(pts0_vis[i][1]))
        p1 = (int(pts1_vis[i][0]) + offset, int(pts1_vis[i][1]))
        c = (int(color[i][2]*255), int(color[i][1]*255), int(color[i][0]*255))
        cv2.line(combined, p0, p1, c, 1)
        cv2.circle(combined, p0, 2, c, -1)
        cv2.circle(combined, p1, 2, c, -1)
        
    avg_conf = np.mean(mconf_np) if n_matches > 0 else 0.0
    text = f"Matches: {n_matches} | Mean Conf: {avg_conf:.4f}"
    cv2.putText(combined, text, (10, TARGET_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(combined, "Reference Map", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "Test Frame", (offset + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return combined, n_matches, avg_conf, mconf_np

if __name__ == "__main__":
    print("\n--- Running LightGlue Confidence Score Comparison ---")
    
    ref_event = os.path.join(BASE, "reference_map.png")
    test_event = os.path.join(BASE, "test_frame_event.png")
    
    # 1. RAW RGB vs RAW Event
    print("\n  MODEL 1: SuperPoint+LightGlue on RAW (RGB Map vs Event Frame)")
    for top_n in [30, 150, 10000]:
        vis_raw, n_raw, conf_raw, confs_raw = run_lightglue(ref_event, test_event, top_n=top_n)
        suffix = "all" if top_n == 10000 else f"{top_n}m"
        cv2.imwrite(os.path.join(BASE, f"comparison_lightglue_event_raw_{suffix}.jpg"), vis_raw)
    
    # 2. Spatial Gradient vs Grayscale Event
    print("\n  MODEL 2: SuperPoint+LightGlue on Spatial Gradient Map vs Grayscale Event")
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

    event_img = cv2.imread(test_event)
    event_gray = cv2.cvtColor(event_img, cv2.COLOR_BGR2GRAY)
    event_gray = cv2.dilate(event_gray, kernel, iterations=1)
    event_grad_path = os.path.join(BASE, "test_frame_event_grad.png")
    cv2.imwrite(event_grad_path, cv2.cvtColor(event_gray, cv2.COLOR_GRAY2BGR))
    
    for top_n in [30, 150, 10000]:
        vis_grad, n_grad, conf_grad, confs_grad = run_lightglue(ref_grad_path, event_grad_path, top_n=top_n)
        suffix = "all" if top_n == 10000 else f"{top_n}m"
        cv2.imwrite(os.path.join(BASE, f"comparison_lightglue_event_grad_{suffix}.jpg"), vis_grad)
    
    # 3. Spatial Gradient vs RAW Event
    print("\n  MODEL 3: SuperPoint+LightGlue on Spatial Gradient Map vs RAW Event Frame")
    for top_n in [30, 150, 10000]:
        vis_grad_raw, n_grad_raw, conf_grad_raw, confs_grad_raw = run_lightglue(ref_grad_path, test_event, top_n=top_n)
        suffix = "all" if top_n == 10000 else f"{top_n}m"
        cv2.imwrite(os.path.join(BASE, f"comparison_lightglue_event_grad_raw_{suffix}.jpg"), vis_grad_raw)
    
    # 4. Grayscale RGB Map vs Grayscale RGB Drone Footage
    print("\n  MODEL 4: SuperPoint+LightGlue on Grayscale RGB Map vs Grayscale Drone Footage")
    test_frame_path = os.path.join(BASE, "test_frame.png")
    test_img = cv2.imread(test_frame_path)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_gray_path = os.path.join(BASE, "scratch_test_gray.png")
    cv2.imwrite(test_gray_path, cv2.cvtColor(test_gray, cv2.COLOR_GRAY2BGR))
    
    ref_gray_path = os.path.join(BASE, "scratch_ref_gray.png")
    cv2.imwrite(ref_gray_path, cv2.cvtColor(ref_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    
    for top_n in [30, 150, 10000]:
        vis_gray, n_gray, conf_gray, confs_gray = run_lightglue(ref_gray_path, test_gray_path, top_n=top_n)
        suffix = "all" if top_n == 10000 else f"{top_n}m"
        cv2.imwrite(os.path.join(BASE, f"comparison_lightglue_gray_drone_{suffix}.jpg"), vis_gray)
    
    print("\n" + "=" * 70)
    print("  LIGHTGLUE CONFIDENCE SCORE RESULTS")
    print("=" * 70)
    print(f"  {'Configuration':<45} {'Confidence':>12} {'Matches':>10}")
    print(f"  {'-'*45} {'-'*12} {'-'*10}")
    print(f"  {'1. RAW RGB Map vs RAW Event Frame':<45} {conf_raw:>12.4f} {n_raw:>10}")
    print(f"  {'2. Gradient Map vs Grayscale Event Frame':<45} {conf_grad:>12.4f} {n_grad:>10}")
    print(f"  {'3. Gradient Map vs RAW Event Frame':<45} {conf_grad_raw:>12.4f} {n_grad_raw:>10}")
    print(f"  {'4. Grayscale Map vs Grayscale Drone Frame':<45} {conf_gray:>12.4f} {n_gray:>10}")
