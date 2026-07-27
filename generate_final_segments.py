"""
SAM 3 Semantic Segmentation - Final Segments
==============================================
Generates two separate segmented images (Map + Frame) using SAM 3
with batch prompting for the macro5 profile.
"""

import os
import cv2
import numpy as np
import torch
import torchvision
from ultralytics.models.sam import SAM3SemanticPredictor

# Monkey-patch torchvision.ops.nms to fix CPU Half precision NotImplementedError
_original_nms = torchvision.ops.nms
def _patched_nms(boxes, scores, iou_threshold):
    return _original_nms(boxes.float(), scores.float(), iou_threshold)
torchvision.ops.nms = _patched_nms

# Paths
BASE = "/home/nalin/gps_denied_project"
REF_PATH  = os.path.join(BASE, "reference_map.png")
TEST_PATH = os.path.join(BASE, "test_frame.png")

OUT_REF  = os.path.join(BASE, "ref_map_seg_final.jpg")
OUT_TEST = os.path.join(BASE, "drone_frame_seg_final.jpg")

# Macro5 Profile
CLASS_DEFS = {
    "building":           {"prompt": "building, house, structure",  "color": (0,0,255),     "conf": 0.15},
    "paved_road":         {"prompt": "road",                       "color": (0,255,255),   "conf": 0.25},
    "unpaved_road":       {"prompt": "unpaved road",               "color": (42,107,168),  "conf": 0.20},
    "tree":               {"prompt": "tree",                       "color": (0,200,0),     "conf": 0.25},
    "water":              {"prompt": "water",                      "color": (255,100,0),   "conf": 0.20},
    "agricultural_field": {"prompt": "agricultural field",         "color": (50,220,50),   "conf": 0.20},
}

def process_image(image_path, out_path, tag):
    """Run SAM 3 on one image and save the segmented overlay."""
    print(f"\n[{tag}] Processing {image_path}...")
    
    # Resize large images to avoid OOM
    MAX_DIM = 1280
    im_orig = cv2.imread(image_path)
    if im_orig is None:
        print(f"  Error: Could not read {image_path}")
        return
        
    h_orig, w_orig = im_orig.shape[:2]
    scale = min(MAX_DIM / max(h_orig, w_orig), 1.0)
    
    if scale < 1.0:
        im_resized = cv2.resize(im_orig, (int(w_orig * scale), int(h_orig * scale)))
        feed_path = image_path + ".tmp_resized.jpg"
        cv2.imwrite(feed_path, im_resized)
        print(f"  Resized {w_orig}x{h_orig} → {im_resized.shape[1]}x{im_resized.shape[0]}")
    else:
        feed_path = image_path

    torch.cuda.empty_cache()

    overrides = dict(
        conf=0.15, task="segment", mode="predict",
        model="sam3.pt", imgsz=320, quantize=16,
        verbose=False, save=False, plots=False, half=False, device=0
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.set_image(feed_path)

    prompts = [v["prompt"] for v in CLASS_DEFS.values()]
    class_names = list(CLASS_DEFS.keys())

    print(f"  Running SAM 3 with {len(prompts)} prompts (Batch Prompting)...")
    results = predictor(text=prompts)

    torch.cuda.empty_cache()
    if scale < 1.0 and os.path.exists(feed_path):
        os.remove(feed_path)

    im = cv2.imread(image_path)
    h, w = im.shape[:2]
    overlay = im.copy()
    alpha = 0.55

    stats = {}
    all_confs = []

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes

        for i, cls_name in enumerate(class_names):
            cls_info = CLASS_DEFS[cls_name]
            color = cls_info["color"]
            cls_conf_thresh = cls_info["conf"]

            if boxes is not None and hasattr(boxes, "cls"):
                cls_indices = np.where(boxes.cls.cpu().numpy() == i)[0]
                if len(cls_indices) == 0:
                    stats[cls_name] = {"count": 0, "mean_conf": 0.0, "area_pct": 0.0}
                    continue

                confs = boxes.conf.cpu().numpy()[cls_indices]
                valid = confs >= cls_conf_thresh
                valid_indices = cls_indices[valid]
                valid_confs = confs[valid]

                if len(valid_indices) == 0:
                    stats[cls_name] = {"count": 0, "mean_conf": 0.0, "area_pct": 0.0}
                    continue

                all_confs.extend(valid_confs.tolist())
                cls_area = 0

                for idx in valid_indices:
                    mask = masks[idx]
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_bool = mask > 0.5
                    cls_area += mask_bool.sum()
                    for c in range(3):
                        overlay[:, :, c] = np.where(
                            mask_bool,
                            ((1 - alpha) * overlay[:, :, c] + alpha * color[c]).astype(np.uint8),
                            overlay[:, :, c],
                        )
                    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, color, 3)

                stats[cls_name] = {
                    "count": len(valid_indices),
                    "mean_conf": float(valid_confs.mean()),
                    "area_pct": round(cls_area / (h * w) * 100, 2),
                }

    # Draw legend
    legend_y = 50
    num_classes = len(CLASS_DEFS)
    box_h = num_classes * 30 + 40
    box_w = 400
    sub_img = overlay[10:10+box_h, 10:10+box_w]
    black_rect = np.zeros_like(sub_img)
    overlay[10:10+box_h, 10:10+box_w] = cv2.addWeighted(sub_img, 0.3, black_rect, 0.7, 0)
    
    cv2.putText(overlay, f"Image: {tag}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    for cls_name, cls_info in CLASS_DEFS.items():
        s = stats.get(cls_name, {"count": 0, "mean_conf": 0.0, "area_pct": 0.0})
        color = cls_info["color"]
        label = f"{cls_name}: {s['count']} ({s['mean_conf']:.2f})"
        
        cv2.rectangle(overlay, (20, legend_y - 14), (36, legend_y), color, -1)
        cv2.rectangle(overlay, (20, legend_y - 14), (36, legend_y), (255, 255, 255), 1)
        cv2.putText(overlay, label, (45, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 30

    overall_conf = float(np.mean(all_confs)) if all_confs else 0.0
    cv2.putText(overlay, f"Overall Conf: {overall_conf:.4f}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(overlay, f"Overall Conf: {overall_conf:.4f}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    cv2.imwrite(out_path, overlay)
    print(f"  Saved segmented image to: {out_path}")

if __name__ == "__main__":
    print("=" * 60)
    print(" SAM 3 Final Segment Generator ")
    print("=" * 60)
    
    process_image(REF_PATH, OUT_REF, "Reference Map")
    process_image(TEST_PATH, OUT_TEST, "Test Frame")
    
    print("\nDONE!")
