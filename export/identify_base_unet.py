"""Identify which SDXL UNet Sentinel's shipped IP-Adapter engine was exported from.

Runs the shipped fp16 `unet_ipadapter` engine (512x896 profile) on fixed random inputs with
ipadapter_scale = 0 (so the 4 image tokens are inert), then runs a candidate PyTorch UNet on the
same inputs and reports the error. A matching base gives max-abs error in the fp16 noise floor
(~1e-2); a different base (Turbo, Lightning, a fine-tune) gives O(1) differences.

Usage: python identify_base_unet.py <engine> <candidate_unet_dir> [<candidate_unet_dir> ...]
"""
import sys
import time

import numpy as np
import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)


def run_engine(path, feeds):
    rt = trt.Runtime(LOGGER)
    with open(path, "rb") as f:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    dev = {}
    out_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            arr = feeds[name]
            t = torch.from_numpy(arr).cuda().contiguous()
            ctx.set_input_shape(name, tuple(arr.shape))
        else:
            out_name = name
            dt = engine.get_tensor_dtype(name)
            tdt = torch.float16 if dt == trt.DataType.HALF else torch.float32
            t = torch.empty(tuple(engine.get_tensor_shape(name)), dtype=tdt, device="cuda")
        dev[name] = t
        ctx.set_tensor_address(name, t.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    assert ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    out = dev[out_name].float().cpu().numpy()
    del ctx, engine, dev
    torch.cuda.empty_cache()
    return out


def main():
    engine_path = sys.argv[1]
    candidates = sys.argv[2:]
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    H, W = 112, 64  # 512x896 profile latent (h x w)
    sample = torch.randn(1, 4, H, W, generator=g).half().numpy()
    timestep = np.array([499.0], dtype=np.float32)
    text = torch.randn(1, 77, 2048, generator=g).half()
    img_tokens = torch.zeros(1, 4, 2048).half()
    ehs81 = torch.cat([text, img_tokens], dim=1).numpy()
    text_embeds = torch.randn(1, 1280, generator=g).half().numpy()
    time_ids = np.array([[896, 512, 0, 0, 896, 512]], dtype=np.float32)
    ipa_scale = np.zeros((70,), dtype=np.float32)

    feeds = {
        "sample": sample,
        "timestep": timestep,
        "encoder_hidden_states": ehs81,
        "text_embeds": text_embeds,
        "time_ids": time_ids,
        "ipadapter_scale": ipa_scale,
    }
    import os
    t0 = time.time()
    if os.path.exists("engine_ref_out.npy") and engine_path == "cached":
        ref = np.load("engine_ref_out.npy")
        print("using cached engine_ref_out.npy", flush=True)
    else:
        ref = run_engine(engine_path, feeds)
        print(f"engine out: shape={ref.shape} mean={ref.mean():.4f} std={ref.std():.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)
        np.save("engine_ref_out.npy", ref)

    from diffusers import UNet2DConditionModel
    for cdir in candidates:
        t0 = time.time()
        unet = UNet2DConditionModel.from_pretrained(cdir, torch_dtype=torch.float16,
                                                    variant="fp16", use_safetensors=True).cuda().eval()
        with torch.no_grad():
            out = unet(
                torch.from_numpy(sample).cuda(),
                torch.tensor([499.0], device="cuda"),
                encoder_hidden_states=text.cuda(),
                added_cond_kwargs={
                    "text_embeds": torch.from_numpy(text_embeds).cuda(),
                    "time_ids": torch.from_numpy(time_ids).cuda().half(),
                },
            ).sample.float().cpu().numpy()
        err = np.abs(out - ref)
        rel = err.mean() / (np.abs(ref).mean() + 1e-6)
        print(f"{cdir}: max_abs={err.max():.4f} mean_abs={err.mean():.4f} rel_mean={rel:.4f} "
              f"corr={np.corrcoef(out.ravel(), ref.ravel())[0,1]:.5f} ({time.time()-t0:.1f}s)",
              flush=True)
        del unet
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
