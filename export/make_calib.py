"""Build an FP8 calibration set for the fused OpticalPattern UNet (Plan v2, Phase 4).

ModelOpt's ONNX post-training quantization needs representative inputs for every graph input.
This writes an .npz with N samples shaped exactly like Sentinel's bindings:

  sample                [N,4,h,w]   f16  noised latents at a spread of timesteps
  timestep              [N]         f32  matching timesteps (heavy at 999 and 650, where the
                                          two-node chain actually runs)
  encoder_hidden_states [N,81,2048] f16  real CLIP embeddings for a prompt bank + 4 zero IPA tokens
  text_embeds           [N,1280]    f16  pooled CLIP embeddings
  time_ids              [N,6]       f32
  controlnet_cond       [N,3,H,W]   f16  three-tone spirals / rings / stripes / checkers
  controlnet_scale      [N]         f32  0.3 .. 1.0
  ipadapter_scale       [N,70]      f32  0 (matches how the chain is run)

x0 latents come from VAE-encoding a mix of real captures (probe outputs, Sentinel captures) and
the control images themselves, so the noised samples cover both scene content and pattern.

Usage:
  python make_calib.py --out calib/realvis_896x512.npz --n 32 [--images dir ...]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_candidate import alphas_cumprod, encode_prompt  # noqa: E402
from export_fused_unet import WEIGHTS  # noqa: E402

VAE_DIR = os.path.join(WEIGHTS, "sdxl-vae-fp16-fix")

PROMPTS = [
    "dense autumn forest, winding dirt path, golden afternoon light, oil painting, highly detailed",
    "cinematic scenery city street, photograph, golden hour, sharp focus",
    "fluffy clouds in a blue sky, highly detailed, 4k",
    "overhead city streets, rush hour, highways, highly detailed, 4k, render",
    "rocky coastline at dusk, long exposure photograph, mist over the water",
    "market square in a medieval town, warm lanterns, wet cobblestones, photograph",
    "snow covered pine forest, low winter sun, film photograph",
    "desert canyon with layered red rock, harsh midday light, photograph",
]

# Timesteps the chain really uses: stage 1 at 999, refine around 550..750, plus a few lower
TIMESTEPS = [999, 999, 999, 899, 749, 699, 649, 599, 549, 449, 349, 249]


def three_tone_patterns(n, W, H, rng):
    """Procedural control images matching OP_Pattern's contract (25% black, 50% grey, 25% white)."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    u = (xs + 0.5) / W - 0.5
    v = (ys + 0.5) / H - 0.5
    aspect = W / H
    out = []
    for i in range(n):
        kind = i % 4
        scale = rng.uniform(1.2, 4.0)
        px, py = u * aspect * scale, v * scale
        phase = rng.uniform(0, 1)
        if kind == 0:  # spiral
            r = np.sqrt(px * px + py * py)
            a = np.arctan2(py, px) / (2 * np.pi)
            val = np.mod(a * rng.integers(2, 6) + np.log(np.maximum(r, 0.002)) * rng.uniform(-1.5, 1.5) - phase, 1.0)
        elif kind == 1:  # rings
            val = np.mod(np.sqrt(px * px + py * py) * 2.0 - phase, 1.0)
        elif kind == 2:  # stripes
            ang = np.deg2rad(rng.uniform(-90, 90))
            val = np.mod(px * np.cos(ang) + py * np.sin(ang) - phase, 1.0)
        else:  # checker
            q = np.floor(px + phase) + np.floor(py + phase)
            f = np.minimum(np.minimum(np.mod(px + phase, 1), 1 - np.mod(px + phase, 1)),
                           np.minimum(np.mod(py + phase, 1), 1 - np.mod(py + phase, 1)))
            inside = (f > 0.15).astype(np.float32)
            val = np.where(np.mod(q, 2) > 0.5, 1.0, 0.0) * inside + 0.5 * (1 - inside)
            out.append(val.astype(np.float32))
            continue
        cov = 0.25
        img = np.full_like(val, 0.5)
        img[val < cov] = 0.0
        img[val > 1 - cov] = 1.0
        out.append(img.astype(np.float32))
    return np.stack(out)  # [n,H,W]


def load_images(paths, W, H):
    ims = []
    for p in paths:
        try:
            ims.append(np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0)
        except Exception:
            pass
    return ims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--images", nargs="*", default=[
        r"C:\Users\times\Sentinel\opticalpattern_build\probes\*.png",
        r"C:\Users\times\Sentinel\workspace\captures\op_exp1*_*.png",
        r"C:\Users\times\Sentinel\workspace\captures\op_exp2*_*.png",
    ])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    W, H = args.width, args.height
    h, w = H // 8, W // 8
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")
    dt = torch.float16

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=dt).to(dev).eval()

    # prompt bank -> CLIP embeddings (text encoders are loaded and freed inside encode_prompt)
    hidden_bank, pooled_bank = [], []
    for p in PROMPTS:
        hdn, pooled = encode_prompt(p, dev)
        hidden_bank.append(hdn[0].float().cpu().numpy())
        pooled_bank.append(pooled[0].float().cpu().numpy())
    print(f"encoded {len(PROMPTS)} prompts", flush=True)

    # x0 sources: real images (probe outputs, captures) plus control patterns
    files = []
    for pat in args.images:
        files += glob.glob(pat)
    files = sorted(set(files))[:48]
    scene = load_images(files, W, H)
    patterns = three_tone_patterns(args.n, W, H, rng)
    print(f"x0 sources: {len(scene)} images, {len(patterns)} patterns", flush=True)

    ac = alphas_cumprod()
    S = {k: [] for k in ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids",
                         "controlnet_cond", "controlnet_scale", "ipadapter_scale"]}
    with torch.no_grad():
        for i in range(args.n):
            t = TIMESTEPS[i % len(TIMESTEPS)]
            a = float(ac[t])
            # x0: mostly scene images, sometimes the pattern itself (what the init looks like at low init_strength)
            if scene and rng.uniform() < 0.8:
                img = scene[i % len(scene)]
            else:
                img = np.repeat(patterns[i][..., None], 3, axis=2)
            x = torch.from_numpy(img).permute(2, 0, 1)[None].to(dev, dt) * 2 - 1
            x0 = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
            noise = torch.randn_like(x0)
            xt = (a ** 0.5) * x0 + ((1 - a) ** 0.5) * noise if t < 999 else noise
            S["sample"].append(xt[0].float().cpu().numpy().astype(np.float16))
            S["timestep"].append(np.float32(t))
            j = i % len(PROMPTS)
            ehs = np.concatenate([hidden_bank[j], np.zeros((4, 2048), np.float32)], axis=0)
            S["encoder_hidden_states"].append(ehs.astype(np.float16))
            S["text_embeds"].append(pooled_bank[j].astype(np.float16))
            S["time_ids"].append(np.array([H, W, 0, 0, H, W], np.float32))
            cond = np.repeat(patterns[i][None], 3, axis=0)
            S["controlnet_cond"].append(cond.astype(np.float16))
            S["controlnet_scale"].append(np.float32(rng.choice([0.3, 0.5, 0.8, 0.85, 1.0])))
            S["ipadapter_scale"].append(np.zeros(70, np.float32))
    arrays = {k: np.stack(v) for k, v in S.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **arrays)
    for k, v in arrays.items():
        print(f"  {k:24} {v.shape} {v.dtype}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
