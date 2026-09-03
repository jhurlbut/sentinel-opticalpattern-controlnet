"""Dump tensor bindings and optimization profiles from Sentinel's shipped TensorRT engines.

Usage: python dump_bindings.py <out.json> <engine1> [<engine2> ...]
Each engine is deserialized in turn (device memory is released between engines).
"""
import json
import os
import sys
import time

import tensorrt as trt

LOGGER = trt.Logger(trt.Logger.WARNING)


def dump(path: str) -> dict:
    t0 = time.time()
    rt = trt.Runtime(LOGGER)
    with open(path, "rb") as f:
        data = f.read()
    engine = rt.deserialize_cuda_engine(data)
    del data
    if engine is None:
        return {"path": path, "error": "deserialize failed"}
    info = {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "num_io_tensors": engine.num_io_tensors,
        "num_optimization_profiles": engine.num_optimization_profiles,
        "num_layers": engine.num_layers,
        "device_memory_size_v2": int(engine.device_memory_size_v2),
        "hardware_compatibility_level": str(engine.hardware_compatibility_level),
        "tensors": [],
    }
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        t = {
            "name": name,
            "mode": "input" if mode == trt.TensorIOMode.INPUT else "output",
            "dtype": str(engine.get_tensor_dtype(name)),
            "shape": list(engine.get_tensor_shape(name)),
            "format": str(engine.get_tensor_format_desc(name)),
            "is_shape_tensor": bool(engine.is_shape_inference_io(name)),
        }
        if mode == trt.TensorIOMode.INPUT:
            profiles = []
            for p in range(engine.num_optimization_profiles):
                try:
                    mn, opt, mx = engine.get_tensor_profile_shape(name, p)
                    profiles.append({"min": list(mn), "opt": list(opt), "max": list(mx)})
                except Exception as e:  # shape tensors / static
                    profiles.append({"error": str(e)})
            t["profiles"] = profiles
        info["tensors"].append(t)
    info["seconds"] = round(time.time() - t0, 1)
    del engine
    return info


def main():
    out = sys.argv[1]
    results = []
    for p in sys.argv[2:]:
        print(f"== {p}", flush=True)
        try:
            r = dump(p)
        except Exception as e:
            r = {"path": p, "error": repr(e)}
        results.append(r)
        for t in r.get("tensors", []):
            prof = t.get("profiles", [{}])[0]
            print(f"  {t['mode']:6} {t['name']:28} {t['dtype']:16} {t['shape']} "
                  f"opt={prof.get('opt')}", flush=True)
        if "error" in r:
            print("  ERROR", r["error"], flush=True)
    with open(out, "w") as f:
        json.dump({"tensorrt": trt.__version__, "engines": results}, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
