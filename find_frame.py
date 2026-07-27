import cv2
import numpy as np
import sys

test_frame = cv2.imread("test_frame.png")
if test_frame is None:
    print("Cannot read test_frame.png")
    sys.exit(1)

test_frame_gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
test_frame_gray = cv2.resize(test_frame_gray, (640, 360)) # downscale for faster matching

videos = [
    "GPSDenied/Videos/Digaru/CameraRecord/VID_EO_14.mp4",
    "GPSDenied/Videos/Digaru/CameraRecord/VID_IR_14.mp4"
]

for vid_path in videos:
    print(f"Searching in {vid_path} ...")
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Could not open {vid_path}")
        continue
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Total Frames: {total_frames}")
    
    best_match_val = float("inf")
    best_frame_idx = -1
    
    # Check every 10th frame to speed up search
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 30 == 0:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.resize(frame_gray, (640, 360))
            
            diff = cv2.absdiff(test_frame_gray, frame_gray)
            mean_diff = np.mean(diff)
            
            if mean_diff < best_match_val:
                best_match_val = mean_diff
                best_frame_idx = frame_idx
                
        frame_idx += 1

    print(f"Best match in {vid_path}: Frame {best_frame_idx} (Time: {best_frame_idx/fps:.2f}s) with diff {best_match_val:.2f}")
    cap.release()
