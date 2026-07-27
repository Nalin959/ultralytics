import torch
import sys
sys.path.append('/home/nalin/gps_denied_project/SP_SLAM3/scripts')
from lightglue import SuperPoint
from lightglue.utils import load_image

img1 = load_image('/home/nalin/gps_denied_project/drone_frame_raw.jpg').cuda()
img2 = load_image('/home/nalin/.gemini/antigravity-ide/brain/b9e5284b-9b2a-4983-958c-ed55daa6eb9b/scratch/frame_test_slam.jpg').cuda()

from lightglue import LightGlue
sp = SuperPoint(max_num_keypoints=2048, keypoint_threshold=0.0005).cuda().eval()
lg = LightGlue(features='superpoint').cuda().eval()

with torch.inference_mode():
    out1 = sp({'image': img1.unsqueeze(0)})
    out2 = sp({'image': img2.unsqueeze(0)})
    matches = lg({'image0': out1, 'image1': out2})
    
print('IMG1 KPTS:', len(out1['keypoints'][0]))
print('IMG2 KPTS:', len(out2['keypoints'][0]))
print('MATCHES:', len(matches['matches'][0]))
