import torch
import tensorrt as trt
import sys
import os

from SP_SLAM3.export_superpoint import SuperPointExport
from SP_SLAM3.scripts.export_lightglue import LightGlueExport

def export_onnx(model, dummy_inputs, output_path, input_names, output_names, dynamic_axes):
    print(f"Exporting {output_path}...")
    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print(f"✅ {output_path} exported successfully.")

def build_engine(onnx_path, engine_path, profile_shapes):
    print(f"Building TensorRT Engine from {onnx_path}...")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
    
    # TRT 11.1 removed platform_has_fast_fp16 and FP16 flag.

    # TRT 11.1 removed EXPLICIT_BATCH because it's the only mode now
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print(f"❌ Failed to parse ONNX file: {onnx_path}")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    profile = builder.create_optimization_profile()
    for name, (min_shape, opt_shape, max_shape) in profile_shapes.items():
        profile.set_shape(name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print(f"❌ Failed to build engine for {onnx_path}")
        return False

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"✅ Saved TensorRT engine to {engine_path}")
    return True

def main():
    device = "cuda"

    # Export SuperPoint
    sp_model = SuperPointExport().eval().to(device)
    sp_dummy = torch.randn(1, 1, 480, 640).to(device)
    export_onnx(
        sp_model,
        sp_dummy,
        "superpoint.onnx",
        input_names=["image"],
        output_names=["prob", "desc"],
        dynamic_axes={
            "image": {2: "height", 3: "width"},
            "prob": {2: "height", 3: "width"},
            "desc": {2: "height", 3: "width"}
        }
    )

    # Export LightGlue
    lg_model = LightGlueExport().eval().to(device)
    lg_dummy = (
        torch.randn(1, 1024, 2).to(device),
        torch.randn(1, 768, 2).to(device),
        torch.randn(1, 1024, 256).to(device),
        torch.randn(1, 768, 256).to(device)
    )
    export_onnx(
        lg_model,
        lg_dummy,
        "lightglue.onnx",
        input_names=["kpts0", "kpts1", "desc0", "desc1"],
        output_names=["matches0", "scores0"],
        dynamic_axes={
            "kpts0": {1: "num_keypoints0"},
            "kpts1": {1: "num_keypoints1"},
            "desc0": {1: "num_keypoints0"},
            "desc1": {1: "num_keypoints1"},
            "matches0": {0: "num_keypoints0"},
            "scores0": {0: "num_keypoints0"}
        }
    )

    # Build Engines
    build_engine("superpoint.onnx", "superpoint.engine", {
        "image": ((1, 1, 128, 128), (1, 1, 480, 640), (1, 1, 1080, 1920))
    })
    
    build_engine("lightglue.onnx", "lightglue.engine", {
        "kpts0": ((1, 1, 2), (1, 1024, 2), (1, 2048, 2)),
        "kpts1": ((1, 1, 2), (1, 1024, 2), (1, 2048, 2)),
        "desc0": ((1, 1, 256), (1, 1024, 256), (1, 2048, 256)),
        "desc1": ((1, 1, 256), (1, 1024, 256), (1, 2048, 256))
    })

if __name__ == "__main__":
    main()
