"""Install a locally built StreamDiff engine as a Sentinel custom pack.

Copies the engine into engines/profiles/sdxl/custom/<name>/<WxH>/ under the loader's
required filename and adds/updates the pack entry in engines/manifest_custom.json.

Loader filename rule (Sentinel 0.5.66):
  tier 0 (IP-Adapter only)      -> unet_ipadapter_fp16.engine
  tier 1 (ControlNet + IPA)     -> unet_controlnet_union_ipadapter_fp16.engine
The non-Union ControlNet filename is resolved through a legacy path and does not load.

Usage:
  python install_custom_pack.py --name glitch --engine engines\glitch_896x512\unet_ipadapter_fp16.engine \
      --display "SDXL Aether Glitch LoRA 896x512" --desc "..." [--controlnet] [--width 896 --height 512]
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil

ENGINES = r"C:\Users\times\Sentinel\engines"
MANIFEST = os.path.join(ENGINES, "manifest_custom.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="profile folder name, e.g. glitch")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--display", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--license", default="see source model page")
    ap.add_argument("--source", default=None, help="source LoRA/checkpoint path for provenance")
    ap.add_argument("--controlnet", action="store_true")
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()

    fname = ("unet_controlnet_union_ipadapter_fp16.engine" if args.controlnet
             else "unet_ipadapter_fp16.engine")
    res = f"{args.width}x{args.height}"
    rel = f"profiles/sdxl/custom/{args.name}/{res}/{fname}"
    dst = os.path.join(ENGINES, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(args.engine)):
        print(f"copying {args.engine} -> {dst}", flush=True)
        shutil.copyfile(args.engine, dst)
    else:
        print(f"engine already in place: {dst}")

    man = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {"packs": {}, "schema_version": 2}
    pack_id = f"custom-sdxl-{args.name}"
    entry = {
        "auto_download": False,
        "base_model": "stabilityai/sdxl-turbo",
        "build_metadata": {
            "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gpu_name": "NVIDIA GeForce RTX 5090 Laptop GPU",
            "sm_capability": "sm120c",
            "tool_version": "opticalpattern_controlnet/export/build_engine.py",
            "trt_version": "10.15.1.29",
        },
        "category": "sdxl",
        "description": args.desc,
        "display_name": args.display,
        "downloads": None,
        "files": [rel],
        "is_local_only": True,
        "license": args.license,
        "pipelines": ["streamdiff"],
        "spec": {"model_name": args.name, "resolution": res},
        "tier": "custom",
    }
    if args.source:
        entry["source_checkpoint"] = {"path": args.source, "sha256": sha256(args.source)}
    man["packs"][pack_id] = entry
    json.dump(man, open(MANIFEST, "w"), indent=2)
    print(f"manifest updated: {pack_id} -> {rel}")
    print(f"engine_profile value: sdxl/custom/{args.name}/{res}   engine_tier: {1 if args.controlnet else 0}")


if __name__ == "__main__":
    main()
