import cv2
import numpy as np
import os
import sys
import torch
import matplotlib.cm as cm

BASE = "/home/nalin/gps_denied_project"

# ----------------- SUPERGLUE SETUP -----------------
sys.path.append(os.path.join(BASE, "SuperGluePretrainedNetwork"))
from models.matching import Matching
from models.utils import frame2tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config_sg = {
    'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 2048},
    'superglue': {'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}
}
matcher_sg = Matching(config_sg).eval().to(device)

def run_sg_color(img0_path, img1_path, top_n, out_path):
    img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    
    max_dim = 1024
    if max(img0.shape) > max_dim:
        scale0 = max_dim / max(img0.shape)
        img0 = cv2.resize(img0, (int(img0.shape[1]*scale0), int(img0.shape[0]*scale0)))
    if max(img1.shape) > max_dim:
        scale1 = max_dim / max(img1.shape)
        img1 = cv2.resize(img1, (int(img1.shape[1]*scale1), int(img1.shape[0]*scale1)))
        
    tensor0 = frame2tensor(img0, device)
    tensor1 = frame2tensor(img1, device)
    
    torch.cuda.empty_cache()
    with torch.inference_mode():
        pred = matcher_sg({'image0': tensor0, 'image1': tensor1})
        
    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()
    
    valid = matches > -1
    pts0 = kpts0[valid]
    pts1 = kpts1[matches[valid]]
    mconf_np = confidence[valid]
    
    if 'scale0' in locals(): pts0 = pts0 / scale0
    if 'scale1' in locals(): pts1 = pts1 / scale1
    
    n_matches = len(pts0)

    img0_cv = cv2.imread(img0_path) # READS IN COLOR!
    img1_cv = cv2.imread(img1_path) # READS IN COLOR!
    
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
        pts0_vis, pts1_vis, mconf_vis = pts0_res[idx], pts1_res[idx], mconf_np[idx]
    else:
        pts0_vis, pts1_vis, mconf_vis = pts0_res, pts1_res, mconf_np
        
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
    cv2.imwrite(out_path, combined)

# ----------------- LIGHTGLUE SETUP -----------------
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher_lg = LightGlue(features='superpoint').eval().to(device)

def run_lg_color(img0_path, img1_path, top_n, out_path):
    image0 = load_image(img0_path).to(device)
    image1 = load_image(img1_path).to(device)
    
    torch.cuda.empty_cache()
    with torch.inference_mode():
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)
        matches01 = matcher_lg({'image0': feats0, 'image1': feats1})
        
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01['matches']
        points0 = feats0['keypoints'][matches[..., 0]]
        points1 = feats1['keypoints'][matches[..., 1]]
        mconf = matches01['scores']

    pts0 = points0.cpu().numpy()
    pts1 = points1.cpu().numpy()
    mconf_np = mconf.cpu().numpy()
    n_matches = len(pts0)

    img0_cv = cv2.imread(img0_path) # READS IN COLOR!
    img1_cv = cv2.imread(img1_path) # READS IN COLOR!
    
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
        pts0_vis, pts1_vis, mconf_vis = pts0_res[idx], pts1_res[idx], mconf_np[idx]
    else:
        pts0_vis, pts1_vis, mconf_vis = pts0_res, pts1_res, mconf_np
        
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
    cv2.imwrite(out_path, combined)

if __name__ == "__main__":
    ref_color = os.path.join(BASE, "reference_map.png")
    test_color = os.path.join(BASE, "test_frame.png")
    
    print("Generating Color Visualizations for Test Frame vs Reference Map")
    
    for top_n in [30, 150, 10000]:
        suffix = "all" if top_n == 10000 else f"{top_n}m"
        
        sg_path = os.path.join(BASE, f"comparison_superglue_color_{suffix}.jpg")
        run_sg_color(ref_color, test_color, top_n, sg_path)
        print(f"Generated {sg_path}")
        
        lg_path = os.path.join(BASE, f"comparison_lightglue_color_{suffix}.jpg")
        run_lg_color(ref_color, test_color, top_n, lg_path)
        print(f"Generated {lg_path}")
