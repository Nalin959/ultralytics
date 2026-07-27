import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
with open("SP_SLAM3/superpoint.engine", "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        dtype = engine.get_binding_dtype(i)
        print(f"Binding {i}: {name}, Shape: {shape}, Type: {dtype}")
