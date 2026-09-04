"""Export a fused SDXL-Turbo UNet (+ optional LoRA) (+ optional OpticalPattern ControlNet)
+ IP-Adapter to ONNX with Sentinel's StreamDiff binding contract (see ../contract/CONTRACT.md).

With ControlNet (profile file unet_controlnet_union_ipadapter_fp16.engine):
  Inputs : sample[1,4,h,w] f16, timestep[1] f32, encoder_hidden_states[1,81,2048] f16,
           text_embeds[1,1280] f16, time_ids[1,6] f32, controlnet_cond[1,3,H,W] f16,
           controlnet_scale[1] f32, ipadapter_scale[70] f32
Without ControlNet (profile file unet_ipadapter_fp16.engine):
  Inputs : sample, timestep, encoder_hidden_states, text_embeds, time_ids, ipadapter_scale
Output   : out_sample[1,4,h,w] f16

Usage:
  python export_fused_unet.py --width 896 --height 512 --out onnx/896x512/fused.onnx
  python export_fused_unet.py --no-controlnet --lora weights/loras/x.safetensors --lora-scale 0.8 \
      --out onnx/896x512_x/unet_ipadapter.onnx
Optional --device cuda|cpu, --dtype fp16|fp32, --validate (ONNX Runtime check; slow on CPU).
"""
import argparse
import gc
import os
import time

import torch
import torch.nn as nn
from safetensors.torch import load_file

BUILD = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = r"C:\Users\times\Sentinel\opticalpattern_build\weights"
TURBO_UNET = os.path.join(WEIGHTS, "sdxl-turbo", "unet")
CN_CONFIG_DIR = os.path.join(WEIGHTS, "controlnet-opticalpattern")
CN_WEIGHTS = r"C:\Users\times\Downloads\sdxlControlnet_v10eOpticalpattern.safetensors"
IPA_WEIGHTS = os.path.join(WEIGHTS, "ip-adapter", "sdxl_models", "ip-adapter_sdxl.safetensors")

N_TEXT = 77
N_IP = 4

INPUT_NAMES = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids",
               "controlnet_cond", "controlnet_scale", "ipadapter_scale"]
INPUT_NAMES_NO_CN = INPUT_NAMES[:5] + INPUT_NAMES[7:]


def input_names(with_controlnet=True):
    return INPUT_NAMES if with_controlnet else INPUT_NAMES_NO_CN


class PreprojectedImageTokens(nn.Module):
    """Stands in for UNet.encoder_hid_proj: Sentinel already projects the CLIP image embeds
    into 4 tokens of width 2048, so the projection must NOT run inside the engine."""

    def forward(self, image_embeds):
        return [image_embeds]


class FusedUNet(nn.Module):
    def __init__(self, unet, controlnet=None):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        # attn2 processors in module order; each gets its own scale entry
        self.ip_procs = [p for n, p in unet.attn_processors.items() if n.endswith("attn2.processor")]
        assert len(self.ip_procs) == 70, f"expected 70 attn2 processors, got {len(self.ip_procs)}"

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids, *rest):
        if self.controlnet is not None:
            controlnet_cond, controlnet_scale, ipadapter_scale = rest
        else:
            (ipadapter_scale,) = rest
        text = encoder_hidden_states[:, :N_TEXT]
        ip_tokens = encoder_hidden_states[:, N_TEXT:N_TEXT + N_IP]
        ipa = ipadapter_scale.to(encoder_hidden_states.dtype)
        for i, proc in enumerate(self.ip_procs):
            proc.scale = [ipa[i]]
        added = {"text_embeds": text_embeds, "time_ids": time_ids.to(text_embeds.dtype)}

        down_res = mid_res = None
        if self.controlnet is not None:
            down_res, mid_res = self.controlnet(
                sample, timestep, encoder_hidden_states=text,
                controlnet_cond=controlnet_cond,
                conditioning_scale=1.0,
                added_cond_kwargs=added, return_dict=False,
            )
            # apply the runtime scale outside the ControlNet so it stays a graph input
            cs = controlnet_scale.to(sample.dtype)
            down_res = [r * cs for r in down_res]
            mid_res = mid_res * cs

        added_unet = dict(added)
        added_unet["image_embeds"] = ip_tokens
        out = self.unet(
            sample, timestep, encoder_hidden_states=text,
            added_cond_kwargs=added_unet,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
            return_dict=False,
        )[0]
        return out


def fuse_lora(unet, lora_path, lora_scale):
    """Kohya or diffusers LoRA -> PEFT adapter on the UNet -> fused into the weights.
    Text-encoder LoRA blocks are dropped: Sentinel's CLIP engines are fixed."""
    from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
    raw = load_file(lora_path)
    # unet_config is required to map SGM-style kohya keys (input_blocks/output_blocks/middle_block)
    # onto diffusers block names; without it those LoRAs fail with 'target modules not found'
    conv = StableDiffusionXLLoraLoaderMixin.lora_state_dict(raw, unet_config=unet.config)
    if isinstance(conv, tuple):
        conv_sd, alphas = conv
    else:
        conv_sd, alphas = conv, None
    n_unet = sum(1 for k in conv_sd if k.startswith("unet."))
    n_te = len(conv_sd) - n_unet
    if n_unet == 0:
        raise SystemExit(f"no UNet LoRA tensors found in {lora_path}")
    # prefix filtering (and stripping) happens inside load_lora_adapter; alphas keep the
    # kohya alpha/rank scaling so the LoRA lands at its trained strength
    unet.load_lora_adapter(conv_sd, network_alphas=alphas, adapter_name="style", prefix="unet")
    unet.fuse_lora(lora_scale=lora_scale, adapter_names=["style"])
    unet.unload_lora()
    print(f"LoRA fused: {os.path.basename(lora_path)} scale={lora_scale} "
          f"unet_tensors={n_unet} dropped_text_encoder_tensors={n_te}", flush=True)


def build_models(device, dtype, lora=None, lora_scale=1.0, with_controlnet=True,
                 base_unet=TURBO_UNET, variant="fp16"):
    from diffusers import ControlNetModel, UNet2DConditionModel
    t0 = time.time()
    kw = {"torch_dtype": dtype, "use_safetensors": True}
    if variant:
        kw["variant"] = variant
    unet = UNet2DConditionModel.from_pretrained(base_unet, **kw)
    print(f"base UNet: {base_unet} (variant={variant})", flush=True)
    if lora:
        fuse_lora(unet, lora, lora_scale)

    # IP-Adapter: installs IPAdapterAttnProcessor2_0 with to_k_ip / to_v_ip on every attn2
    ipa_sd = load_file(IPA_WEIGHTS)
    ip_state = {"image_proj": {}, "ip_adapter": {}}
    for k, v in ipa_sd.items():
        head, rest = k.split(".", 1)
        ip_state[head][rest] = v
    unet._load_ip_adapter_weights([ip_state])
    unet.encoder_hid_proj = PreprojectedImageTokens()
    unet.config.encoder_hid_dim_type = "ip_image_proj"
    unet = unet.to(device=device, dtype=dtype).eval()

    cn = None
    if with_controlnet:
        cn = ControlNetModel.from_config(ControlNetModel.load_config(CN_CONFIG_DIR))
        sd = load_file(CN_WEIGHTS)
        missing, unexpected = cn.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected ControlNet keys: {unexpected[:5]}"
        assert not missing, f"missing ControlNet keys: {missing[:5]}"
        cn = cn.to(device=device, dtype=dtype).eval()
    print(f"models ready in {time.time()-t0:.1f}s", flush=True)
    return FusedUNet(unet, cn).eval()


def make_inputs(width, height, device, dtype, with_controlnet=True):
    h, w = height // 8, width // 8
    g = torch.Generator().manual_seed(0)
    full = (
        torch.randn(1, 4, h, w, generator=g).to(device, dtype),
        torch.tensor([499.0], device=device, dtype=torch.float32),
        torch.randn(1, N_TEXT + N_IP, 2048, generator=g).to(device, dtype),
        torch.randn(1, 1280, generator=g).to(device, dtype),
        torch.tensor([[height, width, 0, 0, height, width]], device=device, dtype=torch.float32),
        torch.rand(1, 3, height, width, generator=g).to(device, dtype),
        torch.tensor([0.7], device=device, dtype=torch.float32),
        torch.full((70,), 0.5, device=device, dtype=torch.float32),
    )
    return full if with_controlnet else full[:5] + full[7:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--lora", default=None, help="LoRA safetensors to fuse into the UNet")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--no-controlnet", action="store_true",
                    help="export a plain IP-Adapter UNet (unet_ipadapter_fp16 profile)")
    ap.add_argument("--base-unet", default=TURBO_UNET,
                    help="diffusers UNet folder to use as the base (default: SDXL-Turbo)")
    ap.add_argument("--variant", default="fp16", help="weights variant, '' for none")
    args = ap.parse_args()
    with_cn = not args.no_controlnet
    names = input_names(with_cn)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    model = build_models(device, dtype, lora=args.lora, lora_scale=args.lora_scale,
                         with_controlnet=with_cn, base_unet=args.base_unet,
                         variant=args.variant or None)
    inputs = make_inputs(args.width, args.height, device, dtype, with_cn)

    with torch.no_grad():
        t0 = time.time()
        ref = model(*inputs)
        print(f"pytorch forward ok: {tuple(ref.shape)} mean={ref.float().mean():.4f} "
              f"std={ref.float().std():.4f} ({time.time()-t0:.1f}s)", flush=True)
        torch.save(ref.float().cpu(), args.out + ".ref.pt")

        t0 = time.time()
        torch.onnx.export(
            model, inputs, args.out,
            input_names=names, output_names=["out_sample"],
            opset_version=args.opset, do_constant_folding=True,
            dynamo=False,
        )
        print(f"onnx export done in {time.time()-t0:.1f}s -> {args.out}", flush=True)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.validate:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        feeds = {n: t.cpu().numpy() for n, t in zip(names, inputs)}
        out = sess.run(["out_sample"], feeds)[0].astype(np.float32)
        r = ref.float().cpu().numpy()
        err = np.abs(out - r)
        print(f"onnxruntime vs pytorch: max_abs={err.max():.4f} mean_abs={err.mean():.5f} "
              f"corr={np.corrcoef(out.ravel(), r.ravel())[0,1]:.6f}", flush=True)


if __name__ == "__main__":
    main()
