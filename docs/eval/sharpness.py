"""Sharpness metrics for comparing engine variants on the same fixture captures.

Two numbers per image, both computed on luminance after resizing to a common width so
engines at different output resolutions are compared fairly:

  lapvar  - variance of the 3x3 Laplacian response. Classic focus measure; higher = more
            fine edge energy. Sensitive to noise, so pair it with the next one.
  hf_ratio - energy of (image - 4x box-downscaled-then-upscaled image) divided by total
            image variance. The fraction of contrast that lives above 1/4 of the sample
            frequency. Robust to global contrast changes.

Usage:
  python sharpness.py --width 896 a.png b.png ...
  python sharpness.py --width 896 --baseline v1.png candidate1.png candidate2.png
With --baseline, ratios against the baseline are printed alongside the raw numbers.
"""
import argparse
import os

import numpy as np
from PIL import Image


def luminance(path, width):
    im = Image.open(path).convert("RGB")
    if width and im.width != width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def laplacian_variance(y):
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    p = np.pad(y, 1, mode="edge")
    out = np.zeros_like(y)
    for dy in range(3):
        for dx in range(3):
            w = k[dy, dx]
            if w:
                out += w * p[dy:dy + y.shape[0], dx:dx + y.shape[1]]
    return float(out.var())


def hf_ratio(y, factor=4):
    h, w = y.shape
    hh, ww = h - h % factor, w - w % factor
    yc = y[:hh, :ww]
    low = yc.reshape(hh // factor, factor, ww // factor, factor).mean(axis=(1, 3))
    low_up = np.repeat(np.repeat(low, factor, axis=0), factor, axis=1)
    total = yc.var()
    return float(((yc - low_up) ** 2).mean() / total) if total > 0 else 0.0


def measure(path, width):
    y = luminance(path, width)
    return laplacian_variance(y), hf_ratio(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--width", type=int, default=896, help="common analysis width (0 = native)")
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args()

    base = measure(args.baseline, args.width) if args.baseline else None
    print(f"{'image':48} {'lapvar':>10} {'hf_ratio':>9}" + ("   vs baseline" if base else ""))
    if base:
        print(f"{os.path.basename(args.baseline)[:48]:48} {base[0]*1e4:10.2f} {base[1]:9.3f}   1.00x  1.00x   (baseline)")
    for p in args.images:
        lv, hf = measure(p, args.width)
        line = f"{os.path.basename(p)[:48]:48} {lv*1e4:10.2f} {hf:9.3f}"
        if base:
            line += f"   {lv/base[0]:.2f}x  {hf/base[1]:.2f}x"
        print(line)
    print("\nlapvar shown x1e4. Compare only captures of the same fixture (same prompt, seed, control image).")


if __name__ == "__main__":
    main()
