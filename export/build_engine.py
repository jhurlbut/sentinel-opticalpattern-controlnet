"""Build a Sentinel-compatible TensorRT engine from the fused ONNX.

Static shapes matching the shipped 896x512 (or 512x896) StreamDiff profile, FP16, one
optimization profile, persistent timing cache. Uses the TensorRT Python builder so no trtexec
binary is required; the tensorrt wheel must match Sentinel's nvinfer_10.dll (10.15.1).

Usage:
  python build_engine.py --onnx onnx/896x512/fused.onnx --width 896 --height 512 \
      --out engines/896x512/unet_controlnet_ipadapter_fp16.engine [--opt-level 4] [--verify]
"""
import argparse
import os
import sys
import time

import numpy as np
import tensorrt as trt

LOGGER = trt.Logger(trt.Logger.INFO)


def shapes_for(width, height):
    h, w = height // 8, width // 8
    return {
        "sample": (1, 4, h, w),
        "timestep": (1,),
        "encoder_hidden_states": (1, 81, 2048),
        "text_embeds": (1, 1280),
        "time_ids": (1, 6),
        "controlnet_cond": (1, 3, height, width),
        "controlnet_scale": (1,),
        "ipadapter_scale": (70,),
    }


def build(onnx_path, out_path, width, height, opt_level, timing_cache, workspace_gb, fp8=False):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, LOGGER)
    t0 = time.time()
    ok = parser.parse_from_file(onnx_path)
    if not ok:
        for i in range(parser.num_errors):
            print("ONNX parse error:", parser.get_error(i), file=sys.stderr)
        raise SystemExit(1)
    print(f"parsed ONNX in {time.time()-t0:.1f}s: {network.num_layers} layers, "
          f"{network.num_inputs} inputs, {network.num_outputs} outputs", flush=True)
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"  input  {t.name:24} {t.dtype} {list(t.shape)}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"  output {t.name:24} {t.dtype} {list(t.shape)}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    if fp8:
        # Q/DQ graph from ModelOpt: let TensorRT use the explicit quantization, keep FP16 elsewhere
        config.set_flag(trt.BuilderFlag.FP8)
        print("FP8 build: explicit Q/DQ quantization enabled", flush=True)
    config.builder_optimization_level = opt_level
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    # Match the shipped engines: no hardware-compat level, one static profile.
    profile = builder.create_optimization_profile()
    present = {network.get_input(i).name for i in range(network.num_inputs)}
    for name, shp in shapes_for(width, height).items():
        if name in present:
            profile.set_shape(name, shp, shp, shp)
    config.add_optimization_profile(profile)

    if timing_cache:
        blob = open(timing_cache, "rb").read() if os.path.exists(timing_cache) else b""
        cache = config.create_timing_cache(blob)
        config.set_timing_cache(cache, ignore_mismatch=False)

    t0 = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise SystemExit("engine build failed")
    print(f"built engine in {(time.time()-t0)/60:.1f} min, {plan.nbytes/1e9:.2f} GB", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(plan)
    if timing_cache:
        with open(timing_cache, "wb") as f:
            f.write(config.get_timing_cache().serialize())
    return out_path


def verify(engine_path, onnx_path, width, height):
    """Run the built engine on the exact seeded inputs used at export time and compare against
    the saved PyTorch reference output (<onnx>.ref.pt). ONNX Runtime is not used: its CUDA
    provider is unreliable in this venv and CPU fp16 is impractical at this size."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from export_fused_unet import input_names, make_inputs

    probe_rt = trt.Runtime(LOGGER)
    probe = probe_rt.deserialize_cuda_engine(open(engine_path, "rb").read())
    names_in_engine = {probe.get_tensor_name(i) for i in range(probe.num_io_tensors)}
    with_cn = "controlnet_cond" in names_in_engine
    del probe
    inputs = make_inputs(width, height, torch.device("cuda"), torch.float16, with_cn)
    feeds = dict(zip(input_names(with_cn), inputs))
    ref = torch.load(onnx_path + ".ref.pt").numpy()

    rt = trt.Runtime(LOGGER)
    engine = rt.deserialize_cuda_engine(open(engine_path, "rb").read())
    ctx = engine.create_execution_context()
    dev = {}
    out_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            t = feeds[name].contiguous()
            ctx.set_input_shape(name, tuple(t.shape))
        else:
            t = torch.empty(tuple(engine.get_tensor_shape(name)), dtype=torch.float16, device="cuda")
            out_name = name
        dev[name] = t
        ctx.set_tensor_address(name, t.data_ptr())
    stream = torch.cuda.Stream()
    assert ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    n = 20
    t0 = time.time()
    for _ in range(n):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    print(f"engine step time: {(time.time()-t0)/n*1000:.1f} ms", flush=True)
    out = dev[out_name].float().cpu().numpy()
    err = np.abs(out - ref)
    print(f"TRT vs PyTorch: max_abs={err.max():.4f} mean_abs={err.mean():.5f} "
          f"rel_mean={err.mean()/(np.abs(ref).mean()+1e-6):.5f} "
          f"corr={np.corrcoef(out.ravel(), ref.ravel())[0,1]:.6f}", flush=True)
    del ctx, engine
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--opt-level", type=int, default=4)
    ap.add_argument("--workspace-gb", type=float, default=16.0)
    ap.add_argument("--timing-cache", default="opticalpattern_timing.cache")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--fp8", action="store_true", help="build a ModelOpt Q/DQ graph with the FP8 flag")
    args = ap.parse_args()
    print("TensorRT", trt.__version__, flush=True)
    if not args.skip_build:
        build(args.onnx, args.out, args.width, args.height, args.opt_level, args.timing_cache,
              args.workspace_gb, fp8=args.fp8)
    if args.verify:
        verify(args.out, args.onnx, args.width, args.height)


if __name__ == "__main__":
    main()
