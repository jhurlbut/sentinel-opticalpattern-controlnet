"""Build a diagnostic 'engine' that lets Sentinel reveal its denoise -> timestep mapping.

The engine has exactly the ControlNet+IP-Adapter binding contract but predicts
eps = sample * 1e-3 (numerically ~zero). Whatever update Sentinel applies after the step, an
epsilon of ~0 at timestep t turns x_t = sqrt(a_t) * init + sqrt(1 - a_t) * noise into
x0 = x_t / sqrt(a_t) = init + noise * sqrt((1 - a_t) / a_t). With a known flat init (the
three-tone control image wired into Video Input, feedback 0) the noise amplitude in the flat
mid-grey region measures a_t, hence t, for each denoise value.

Usage:
  python build_timestep_probe.py --width 896 --height 512 --out engines/probe_896x512/unet_controlnet_ipadapter_fp16.engine
Then install with install_custom_pack.py --name tprobe --controlnet, set the node to that
profile, and capture at denoise 1.0 / 0.75 / 0.5 / 0.25 with the control image as Video Input.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_engine import build  # noqa: E402
from export_fused_unet import INPUT_NAMES, make_inputs  # noqa: E402


class Probe(nn.Module):
    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids,
                controlnet_cond, controlnet_scale, ipadapter_scale):
        # Touch every input with a negligible weight so TensorRT keeps all bindings.
        tiny = (encoder_hidden_states.float().mean() + text_embeds.float().mean()
                + time_ids.float().mean() + controlnet_cond.float().mean()
                + controlnet_scale.float().mean() + ipadapter_scale.float().mean()
                + timestep.float().mean()) * 1e-12
        return (sample.float() * 1e-3 + tiny).to(sample.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    onnx_path = os.path.splitext(args.out)[0] + ".onnx"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    inputs = make_inputs(args.width, args.height, torch.device("cpu"), torch.float16, True)
    torch.onnx.export(Probe().eval(), inputs, onnx_path, input_names=INPUT_NAMES,
                      output_names=["out_sample"], opset_version=17, dynamo=False)
    build(onnx_path, args.out, args.width, args.height, opt_level=1, timing_cache=None, workspace_gb=1.0)
    print("probe engine:", args.out)


if __name__ == "__main__":
    main()
