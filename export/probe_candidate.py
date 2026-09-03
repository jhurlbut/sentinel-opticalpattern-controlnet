"""Phase 3 gate: render one-step images from a candidate base UNet with the OpticalPattern
ControlNet, using the same call shape Sentinel's engine sees, before spending a TensorRT build.

For each requested timestep it starts from pure noise (denoise 1.0), runs ONE UNet call with the
ControlNet residuals, converts the epsilon prediction to x0 with the standard SDXL alphas, and
decodes with the fp16-fix VAE. If the candidate cannot make a clean image this way, Sentinel
cannot either, whatever the build says.

Usage:
  python probe_candidate.py --unet weights/realvisxl-v5/unet --lora weights/dmd2/dmd2_sdxl_4step_lora_fp16.safetensors \
      --control <spiral control png> --prompt "dense autumn forest, winding dirt path" \
      --timesteps 999 749 499 399 --cn-scale 0.8 --out probes/realvis_dmd2

Produces <out>_t<timestep>.png per timestep plus a contact sheet <out>_sheet.png.
Turbo baseline for comparison: --unet weights/sdxl-turbo/unet --variant fp16
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_fused_unet import CN_CONFIG_DIR, CN_WEIGHTS, IPA_WEIGHTS, WEIGHTS, PreprojectedImageTokens, fuse_lora  # noqa: E402

TURBO = os.path.join(WEIGHTS, "sdxl-turbo")
VAE_DIR = os.path.join(WEIGHTS, "sdxl-vae-fp16-fix")


def encode_prompt(prompt, device):
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    tok1 = CLIPTokenizer.from_pretrained(os.path.join(TURBO, "tokenizer"))
    tok2 = CLIPTokenizer.from_pretrained(os.path.join(TURBO, "tokenizer_2"))
    te1 = CLIPTextModel.from_pretrained(os.path.join(TURBO, "text_encoder"), torch_dtype=torch.float16,
                                        variant="fp16", use_safetensors=True).to(device)
    te2 = CLIPTextModelWithProjection.from_pretrained(os.path.join(TURBO, "text_encoder_2"), torch_dtype=torch.float16,
                                                      variant="fp16", use_safetensors=True).to(device)
    with torch.no_grad():
        ids1 = tok1(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
        ids2 = tok2(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
        o1 = te1(ids1, output_hidden_states=True)
        o2 = te2(ids2, output_hidden_states=True)
        hidden = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)  # [1,77,2048]
        pooled = o2.text_embeds  # [1,1280]
    del te1, te2
    torch.cuda.empty_cache()
    return hidden, pooled


def alphas_cumprod():
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
    return torch.cumprod(1.0 - betas, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", required=True)
    ap.add_argument("--variant", default=None, help="fp16 for the Turbo folder, omit for RealVisXL")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--control", required=True, help="control image png (896x512 three-tone)")
    ap.add_argument("--prompt", default="dense autumn forest, winding dirt path, golden afternoon light, highly detailed")
    ap.add_argument("--timesteps", type=int, nargs="+", default=[999, 749, 499, 399])
    ap.add_argument("--cn-scale", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
    device = torch.device("cuda")
    dtype = torch.float16
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    hidden, pooled = encode_prompt(args.prompt, device)

    kw = {"torch_dtype": dtype, "use_safetensors": True}
    if args.variant:
        kw["variant"] = args.variant
    unet = UNet2DConditionModel.from_pretrained(args.unet, **kw)
    if args.lora:
        fuse_lora(unet, args.lora, args.lora_scale)
    # IP-Adapter present (as in the engine) but at scale 0 so the probe isolates base + ControlNet
    ipa_sd = load_file(IPA_WEIGHTS)
    ip_state = {"image_proj": {}, "ip_adapter": {}}
    for k, v in ipa_sd.items():
        head, rest = k.split(".", 1)
        ip_state[head][rest] = v
    unet._load_ip_adapter_weights([ip_state])
    unet.encoder_hid_proj = PreprojectedImageTokens()
    unet.config.encoder_hid_dim_type = "ip_image_proj"
    for n, p in unet.attn_processors.items():
        if n.endswith("attn2.processor"):
            p.scale = [0.0]
    unet = unet.to(device, dtype).eval()

    cn = ControlNetModel.from_config(ControlNetModel.load_config(CN_CONFIG_DIR))
    cn.load_state_dict(load_file(CN_WEIGHTS), strict=True)
    cn = cn.to(device, dtype).eval()
    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=dtype).to(device).eval()

    ctrl = Image.open(args.control).convert("RGB")
    W, H = ctrl.size
    cond = torch.from_numpy(np.asarray(ctrl, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device, dtype)
    ip_tokens = torch.zeros(1, 4, 2048, device=device, dtype=dtype)
    added = {"text_embeds": pooled, "time_ids": torch.tensor([[H, W, 0, 0, H, W]], device=device, dtype=dtype),
             "image_embeds": ip_tokens}
    ac = alphas_cumprod()

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(1, 4, H // 8, W // 8, generator=g).to(device, dtype)
    tiles = []
    for t in args.timesteps:
        a = ac[t].item()
        x_t = noise  # denoise 1.0: pure noise, scaled like a fully noised latent (x0 term vanishes)
        ts = torch.tensor([float(t)], device=device)
        with torch.no_grad():
            down, mid = cn(x_t, ts, encoder_hidden_states=hidden, controlnet_cond=cond, conditioning_scale=1.0,
                           added_cond_kwargs={"text_embeds": pooled, "time_ids": added["time_ids"]}, return_dict=False)
            down = [d * args.cn_scale for d in down]
            mid = mid * args.cn_scale
            eps = unet(x_t, ts, encoder_hidden_states=hidden, added_cond_kwargs=added,
                       down_block_additional_residuals=down, mid_block_additional_residual=mid, return_dict=False)[0]
            x0 = (x_t.float() - (1 - a) ** 0.5 * eps.float()) / a ** 0.5
            img = vae.decode((x0 / vae.config.scaling_factor).to(dtype), return_dict=False)[0]
        img = ((img[0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        im = Image.fromarray(img)
        im.save(f"{args.out}_t{t}.png")
        tiles.append((t, im))
        print(f"t={t}: alpha_bar={a:.4f} x0 std={x0.std():.3f} -> {args.out}_t{t}.png", flush=True)

    tw, th = 448, round(448 * H / W)
    sheet = Image.new("RGB", (len(tiles) * (tw + 8) + 8, th + 36), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    for i, (t, im) in enumerate(tiles):
        x = 8 + i * (tw + 8)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, 28))
        d.text((x + 2, 8), f"t={t}", fill=(220, 220, 220))
    sheet.save(f"{args.out}_sheet.png")
    print("sheet:", f"{args.out}_sheet.png")


if __name__ == "__main__":
    main()
