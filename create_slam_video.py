import cv2
import os

input_video = 'test_slam.mp4'
output_video = 'slam_features_video.mp4'

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"Failed to open {input_video}")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

orb = cv2.ORB_create(
    nfeatures=1500,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=7
)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    keypoints = orb.detect(frame, None)
    img_with_kp = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
    
    # Add text overlay
    cv2.putText(img_with_kp, f"ORB Features: {len(keypoints)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
    out.write(img_with_kp)
    frame_count += 1
    
    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print(f"Finished writing {frame_count} frames to {output_video}")
