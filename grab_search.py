import cv2
import numpy as np
import sys

test_frame = cv2.imread("test_frame.png")
test_gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
test_gray = cv2.resize(test_gray, (640, 360))

vid_path = "GPSDenied/Videos/Digaru/CameraRecord/VID_EO_14.mp4"
cap = cv2.VideoCapture(vid_path)
fps = cap.get(cv2.CAP_PROP_FPS)

best_diff = float('inf')
best_frame = -1

frame_idx = 0
step = 150 # Every 5 seconds roughly (if 30 fps)

while True:
    ret = cap.grab()
    if not ret: break
    
    if frame_idx % step == 0:
        ret, frame = cap.retrieve()
        if not ret: break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.resize(frame_gray, (640, 360))
        diff = np.mean(cv2.absdiff(test_gray, frame_gray))
        
        if diff < best_diff:
            best_diff = diff
            best_frame = frame_idx
            print(f"New best at frame {frame_idx} (time: {frame_idx/fps:.2f}s) diff: {diff:.2f}")
            
    frame_idx += 1

print(f"Final Coarse best match: Frame {best_frame} at {best_frame/fps:.2f}s")
cap.release()
