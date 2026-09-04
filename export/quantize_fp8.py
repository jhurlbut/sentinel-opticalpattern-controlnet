"""Quantize a fused OpticalPattern UNet ONNX to FP8 with NVIDIA ModelOpt (Plan v2, Phase 4).

Inserts Q/DQ nodes around Conv/MatMul/Gemm using the calibration set from make_calib.py and
writes a new ONNX (external data). Build it with build_engine.py --fp8. The ControlNet branch
can be kept in FP16 with --keep-controlnet-fp16 if the quantized engine loses the figure.

Usage:
  python quantize_fp8.py --onnx onnx/realvis_896x512/fused.onnx --calib calib/realvis_896x512.npz \
      --out onnx/realvis_896x512_fp8/fused_fp8.onnx [--keep-controlnet-fp16]
"""
import argparse
import os
import shutil
import time

import numpy as np
import onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="fp8", help="fp8 (default) or int8")
    ap.add_argument("--keep-controlnet-fp16", action="store_true",
                    help="exclude nodes whose name contains 'controlnet' from quantization")
    ap.add_argument("--eps", default="cuda:0,cpu", help="calibration execution providers, comma separated")
    ap.add_argument("--calibration-method", default="max",
                    help="max (default, robust for fp8) | entropy | percentile | distribution")
    args = ap.parse_args()

    from modelopt.onnx.quantization import quantize
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    calib = dict(np.load(args.calib))
    n = next(iter(calib.values())).shape[0]
    print(f"calibration set: {n} samples, inputs: {list(calib)}", flush=True)

    # Sentinel's contract has batch-less inputs (timestep [1], controlnet_scale [1],
    # ipadapter_scale [70]); ModelOpt's default provider splits every array on axis 0 and would
    # hand the model a [1,70]. Feed exact per-sample shapes ourselves instead.
    NO_BATCH = {"ipadapter_scale"}

    class SentinelReader(CalibrationDataReader):
        def __init__(self, data):
            self.samples = []
            for i in range(n):
                feed = {}
                for k, v in data.items():
                    feed[k] = v[i] if k in NO_BATCH else v[i:i + 1]
                self.samples.append(feed)
            self.rewind()

        def get_next(self):
            return next(self.it, None)

        def get_first(self):
            return self.samples[0]

        def rewind(self):
            self.it = iter(self.samples)

    reader = SentinelReader(calib)
    print("per-sample shapes:", {k: tuple(v.shape) for k, v in reader.get_first().items()}, flush=True)

    nodes_to_exclude = None
    if args.keep_controlnet_fp16:
        m = onnx.load(args.onnx, load_external_data=False)
        nodes_to_exclude = [nd.name for nd in m.graph.node if "controlnet" in nd.name.lower()]
        print(f"keeping {len(nodes_to_exclude)} ControlNet nodes in FP16", flush=True)
        del m

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    quantize(
        onnx_path=args.onnx,
        quantize_mode=args.mode,
        calibration_data_reader=reader,
        calibration_method=args.calibration_method,
        calibration_eps=args.eps.split(","),
        nodes_to_exclude=nodes_to_exclude,
        use_external_data_format=True,
        high_precision_dtype="fp16",
        mha_accumulation_dtype="fp16",
        output_path=args.out,
        log_level="INFO",
    )
    print(f"quantized in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)
    # keep the FP16 reference next to the quantized graph so build_engine --verify can compare
    ref = args.onnx + ".ref.pt"
    if os.path.exists(ref):
        shutil.copyfile(ref, args.out + ".ref.pt")


if __name__ == "__main__":
    main()
