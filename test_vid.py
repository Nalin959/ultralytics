import cv2
import sys
cap = cv2.VideoCapture("test_slam.mp4")
if not cap.isOpened():
    print("Cannot open")
    sys.exit(1)
ret, frame = cap.read()
if not ret:
    print("Cannot read frame")
else:
    print(f"Read frame of shape {frame.shape}")
