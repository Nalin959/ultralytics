import tensorrt as trt
import numpy as np
import torch
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image
import cv2

device = torch.device('cuda')
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

def normalize(kpts, image):
    h, w = image.shape[2], image.shape[3]
    size = max(h, w)
    kpts_norm = kpts.clone()
    kpts_norm[..., 0] = (kpts[..., 0] - w/2) / (size/2)
    kpts_norm[..., 1] = (kpts[..., 1] - h/2) / (size/2)
    return kpts_norm

kpts0 = normalize(feats0['keypoints'], image0).contiguous()
desc0 = feats0['descriptors'].contiguous()
kpts1 = normalize(feats1['keypoints'], image1).contiguous()
desc1 = feats1['descriptors'].contiguous()

logger = trt.Logger(trt.Logger.WARNING)
with open("lightglue.engine", "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    context.set_input_shape("kpts0", kpts0.shape)
    context.set_input_shape("desc0", desc0.shape)
    context.set_input_shape("kpts1", kpts1.shape)
    context.set_input_shape("desc1", desc1.shape)

    matches0 = torch.zeros((kpts0.shape[1],), dtype=torch.int64, device='cuda')
    scores0 = torch.zeros((kpts0.shape[1],), dtype=torch.float32, device='cuda')

    context.set_tensor_address("kpts0", kpts0.data_ptr())
    context.set_tensor_address("desc0", desc0.data_ptr())
    context.set_tensor_address("kpts1", kpts1.data_ptr())
    context.set_tensor_address("desc1", desc1.data_ptr())
    context.set_tensor_address("matches0", matches0.data_ptr())
    context.set_tensor_address("scores0", scores0.data_ptr())

    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)

    valid = matches0 > -1
    matches = matches0[valid]
    scores = scores0[valid]
    print(f"TRT Python Matches: {len(matches)}, Mean Conf: {scores.mean().item():.4f}")

