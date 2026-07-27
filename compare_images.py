import cv2
import numpy as np

img1 = cv2.imread("test_frame.png")
img2 = cv2.imread("GPSDenied/Videos/Digaru/CameraRecord/IMG_EO_00016.jpeg")

if img1 is None or img2 is None:
    print("Cannot read images")
    exit(1)

# resize if different
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

diff = np.mean(cv2.absdiff(img1, img2))
print(f"Diff between test_frame.png and IMG_EO_00016.jpeg: {diff:.2f}")

img3 = cv2.imread("GPSDenied/Videos/Digaru/CameraRecord/IMG_IR_00016.jpeg")
if img3 is not None:
    print(f"IMG_IR_00016.jpeg shape: {img3.shape}")

