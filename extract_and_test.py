import cv2
import sys

# 1. Extract the last frame
cap = cv2.VideoCapture("v2e_out_ir/dvs-video.avi")
last_frame = None
while True:
    ret, frame = cap.read()
    if not ret: break
    last_frame = frame
cap.release()

if last_frame is None:
    print("Could not extract frame from dvs-video.avi")
    sys.exit(1)

cv2.imwrite("test_frame_ir_event.png", last_frame)
print("Saved test_frame_ir_event.png")
