import cv2
import numpy as np
import os
import gc
import torch
from compare_sam3_vs_loftr import run_sam3, run_loftr

in_video = "v2e_out_ir/dvs-video.avi"
out_video = "final_output.mp4"
ref_img = "reference_map.png"
temp_test = "temp_frame.png"

cap = cv2.VideoCapture(in_video)
fps = cap.get(cv2.CAP_PROP_FPS)

# We will process every 4th frame (25 fps effectively from 100fps)
frame_skip = 4
out_fps = fps / frame_skip

out = None

frame_idx = 0
processed = 0

print(f"Starting video processing. Output will be ~{out_fps} fps.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    if frame_idx % frame_skip == 0:
        print(f"Processing frame {frame_idx} (processed {processed})...")
        cv2.imwrite(temp_test, frame)
        
        # SAM3 returns (overlay, stats, conf)
        # overlay is original image size (640x512)
        sam3_overlay, _, sam3_conf = run_sam3(temp_test, tag=f"Frame {frame_idx}")
        
        # LoFTR returns (canvas_bgr, n_matches, mean_conf, mconf)
        # canvas_bgr is (720, nw0 + nw1, 3) where it's stitched side-by-side
        loftr_canvas, _, loftr_conf, _ = run_loftr(ref_img, temp_test)
        
        target_h = loftr_canvas.shape[0]
        
        # resize sam3_overlay to match height
        s_scale = target_h / sam3_overlay.shape[0]
        nw_sam3 = int(sam3_overlay.shape[1] * s_scale)
        sam3_resized = cv2.resize(sam3_overlay, (nw_sam3, target_h))
        
        # add a title to sam3
        cv2.putText(sam3_resized, "SAM3 Segmented", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.putText(sam3_resized, "SAM3 Segmented", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # concatenate horizontally: sam3_resized + loftr_canvas
        final_frame = np.hstack((sam3_resized, loftr_canvas))
        
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w = final_frame.shape[:2]
            out = cv2.VideoWriter(out_video, fourcc, out_fps, (w, h))
            
        out.write(final_frame)
        processed += 1
        gc.collect()
        torch.cuda.empty_cache()
        
    frame_idx += 1

cap.release()
if out is not None:
    out.release()
print("Video processing complete!")

# Clean up temp file
if os.path.exists(temp_test):
    os.remove(temp_test)
