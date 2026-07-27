import torch
import cv2
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image

device = torch.device('cuda')
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features='superpoint').eval().to(device)

image0 = load_image('test_frame_gray.png').cuda()
image1 = load_image('reference_map_event.png').cuda()

feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)
matches01 = matcher({'image0': feats0, 'image1': feats1})

print(f"Keypoints 0: {feats0['keypoints'].shape[1]}")
print(f"Keypoints 1: {feats1['keypoints'].shape[1]}")
print(f"Matches: {matches01['matches'][0].shape[0]}")
