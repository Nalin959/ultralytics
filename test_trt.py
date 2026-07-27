import tensorrt as trt
import numpy as np
import torch
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image

device = torch.device('cuda')
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

image0 = load_image('test_frame_gray.png').cuda()
image1 = load_image('reference_map_event.png').cuda()

feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)

def normalize(kpts, image):
    h, w = image.shape[1], image.shape[2]
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

    print("desc0 sum:", desc0.abs().sum().item(), "desc1 sum:", desc1.abs().sum().item())
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)

    print("TRT Matches:", (matches0 > -1).sum().item())

