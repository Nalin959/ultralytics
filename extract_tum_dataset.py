import cv2
import os

video_path = "test_clip.mp4"
output_dir = "test_sequence"
rgb_dir = os.path.join(output_dir, "rgb")

os.makedirs(rgb_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video Resolution: {width}x{height}, FPS: {fps}")

rgb_txt_path = os.path.join(output_dir, "rgb.txt")

with open(rgb_txt_path, "w") as f:
    f.write("# color images\n")
    f.write("# file: 'test_clip.mp4'\n")
    f.write("# timestamp filename\n")
    
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = frame_count / fps
        filename = f"{timestamp:.6f}.png"
        filepath = os.path.join(rgb_dir, filename)
        
        cv2.imwrite(filepath, frame)
        
        f.write(f"{timestamp:.6f} rgb/{filename}\n")
        saved_count += 1
            
        frame_count += 1

cap.release()
print(f"Extracted {saved_count} frames to {output_dir}")
