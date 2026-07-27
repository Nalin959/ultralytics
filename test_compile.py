import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1 << 30))

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

onnx_path = "lightglue.onnx"
with open(onnx_path, "rb") as model:
    if not parser.parse(model.read()):
        print(f"Failed to parse {onnx_path}")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
    else:
        print(f"Successfully parsed {onnx_path}! Building engine...")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes:
            print("Successfully built TensorRT engine!")
        else:
            print("Failed to build TensorRT engine.")
