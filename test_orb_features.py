import cv2
import numpy as np

# Load test frame
img = cv2.imread('test_sequence/rgb/4.001945.png')
if img is None:
    print("Failed to load test_sequence/rgb/4.001945.png")
    exit(1)

# Initialize ORB detector similar to ORB-SLAM3 settings (1000 features, 8 levels)
orb = cv2.ORB_create(
    nfeatures=1000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=7
)

# Find keypoints
keypoints = orb.detect(img, None)

# Draw keypoints (green dots)
img_with_kp = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)

# Save result
cv2.imwrite('orb_features_test_slam.jpg', img_with_kp)
print(f"Extracted {len(keypoints)} ORB features.")
