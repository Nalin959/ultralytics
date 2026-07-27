import torch
import numpy as np
import cv2
from lightglue import SuperPoint, LightGlue

device = torch.device('cpu')
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

def load_clahe_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = torch.from_numpy(img).float()[None, None] / 255.0
    return img.to(device)

image0 = load_clahe_image('test_frame.png')
image1 = load_clahe_image('reference_map.png')

feats0 = extractor({'image': image0})
feats1 = extractor({'image': image1})

# Test full 9 layers
matcher9 = LightGlue(features='superpoint', depth=9, depth_confidence=-1, width_confidence=-1).eval().to(device)
pred9 = matcher9({'image0': feats0, 'image1': feats1})
scores9 = pred9['matching_scores0'][0]
scores9 = scores9[pred9['matches0'][0] > -1]

# Test 4 layers
matcher4 = LightGlue(features='superpoint', depth=4, depth_confidence=-1, width_confidence=-1).eval().to(device)
pred4 = matcher4({'image0': feats0, 'image1': feats1})
scores4 = pred4['matching_scores0'][0]
scores4 = scores4[pred4['matches0'][0] > -1]

print(f"9 Layers - Matches: {len(scores9)}, Mean Conf: {scores9.mean().item():.4f}")
print(f"4 Layers - Matches: {len(scores4)}, Mean Conf: {scores4.mean().item():.4f}")
