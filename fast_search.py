import cv2
import numpy as np
import sys
import math

test_frame = cv2.imread("test_frame.png")
test_gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
test_gray = cv2.resize(test_gray, (640, 360))

vid_path = "GPSDenied/Videos/Digaru/CameraRecord/VID_EO_14.mp4"
cap = cv2.VideoCapture(vid_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"Video duration: {duration:.2f}s")

def get_frame_diff(time_sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
    ret, frame = cap.read()
    if not ret: return float('inf')
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.resize(frame_gray, (640, 360))
    return np.mean(cv2.absdiff(test_gray, frame_gray))

# Step 1: Coarse search every 10 seconds
best_diff = float('inf')
best_t = 0
for t in range(0, int(duration), 5):
    diff = get_frame_diff(t)
    if diff < best_diff:
        best_diff = diff
        best_t = t

print(f"Coarse best match at {best_t}s (diff {best_diff:.2f})")

# Step 2: Fine search around best_t +/- 5s, every 0.5s
best_fine_t = best_t
for t_offset in np.arange(-5, 5, 0.5):
    t = best_t + t_offset
    if t < 0 or t > duration: continue
    diff = get_frame_diff(t)
    if diff < best_diff:
        best_diff = diff
        best_fine_t = t

print(f"Fine best match at {best_fine_t:.2f}s (diff {best_diff:.2f})")

# Step 3: Frame-by-frame search around best_fine_t +/- 0.5s
best_frame_t = best_fine_t
for t_offset in np.arange(-0.5, 0.5, 1.0/fps):
    t = best_fine_t + t_offset
    if t < 0 or t > duration: continue
    diff = get_frame_diff(t)
    if diff < best_diff:
        best_diff = diff
        best_frame_t = t

print(f"Exact best match at {best_frame_t:.2f}s (diff {best_diff:.2f})")
cap.release()
