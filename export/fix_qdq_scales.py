"""Patch non-positive Q/DQ scales in a ModelOpt-quantized ONNX so TensorRT accepts it.

Tensors that were identically zero during calibration (the IP-Adapter branch when the
calibration set runs with ipadapter_scale 0) get a scale of 0, which TensorRT rejects. Any
positive scale is exact for a zero tensor, so set those to 1.0. Tiny positive scales are left.

Usage: python fix_qdq_scales.py --onnx onnx/realvis_896x512_fp8/fused_fp8.onnx
"""
import argparse
import os
import time

import numpy as np
import onnx
from onnx import numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    args = ap.parse_args()
    t0 = time.time()
    model = onnx.load(args.onnx)  # loads external data too
    inits = {i.name: i for i in model.graph.initializer}
    scale_names = set()
    for nd in model.graph.node:
        if nd.op_type in ("QuantizeLinear", "DequantizeLinear") and len(nd.input) > 1:
            scale_names.add(nd.input[1])
    fixed = 0
    checked = 0
    for name in scale_names:
        init = inits.get(name)
        if init is None:
            continue
        checked += 1
        arr = numpy_helper.to_array(init)
        bad = ~(arr > 0)
        if bad.any():
            new = arr.astype(np.float32).copy()
            new[bad] = 1.0
            init.CopyFrom(numpy_helper.from_array(new.astype(arr.dtype), name))
            fixed += 1
    print(f"scale initializers checked: {checked}, patched: {fixed} ({time.time()-t0:.0f}s)", flush=True)
    if fixed:
        d = os.path.dirname(os.path.abspath(args.onnx))
        loc = os.path.basename(args.onnx) + ".data"
        # remove old external files so the save is clean
        for f in os.listdir(d):
            if f != os.path.basename(args.onnx) and not f.endswith(".ref.pt"):
                os.remove(os.path.join(d, f))
        onnx.save_model(model, args.onnx, save_as_external_data=True, all_tensors_to_one_file=True,
                        location=loc, size_threshold=1024)
        print("saved", args.onnx, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
