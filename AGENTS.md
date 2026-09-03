# Agent Guide: OpticalPattern ControlNet for Sentinel StreamDiff

This repository is a worked example of bringing a third-party SDXL ControlNet into Sentinel's
real-time StreamDiff node as a custom TensorRT engine pack, plus the Sentinel project that
drives it. Read this file before changing anything.

## What is here

| Path | Role |
|---|---|
| `opticalpattern_controlnet.sentinel` | Sentinel project: Camera -> OP_Pattern -> OP_Diffusion (StreamDiff) |
| `modules/OP_Pattern/` | Control-image generator Module (HLSL, three passes, one stat buffer) |
| `modules/_shared/plan_theme.hlsli` | Instrument palette used by the Monitor output |
| `contract/` | Sentinel's StreamDiff engine binding contract, dumped from the shipped engine |
| `export/` | ONNX export, TensorRT build, engine verification, pack installer, LoRA batch |
| `docs/` | Proof images, tuning notes, correspondence draft |
| `PLAN.md` | The phased plan that produced this repo |

Build artifacts (Python venv, downloaded weights, ONNX, engines, timing cache) are deliberately
outside the repo. The scripts default to `C:\Users\<user>\Sentinel\opticalpattern_build\`;
change the `WEIGHTS` constant in `export/export_fused_unet.py` if you put them elsewhere.

## Non-negotiables

1. **The binding contract is fixed by Sentinel, not by you.** Input names, dtypes, shapes and
   order are in `contract/CONTRACT.md`. An engine that deviates loads silently and produces
   garbage or crashes the node. Re-dump the contract with `contract/dump_bindings.py` when
   targeting a different Sentinel version.
2. **Engine filename rule.** A ControlNet engine must be installed as
   `unet_controlnet_union_ipadapter_fp16.engine` even when it is a single-type ControlNet.
   An IP-Adapter-only engine is `unet_ipadapter_fp16.engine`. Other names resolve through a
   legacy path and never load.
3. **TensorRT plans are GPU-architecture specific.** An engine built on an RTX 50-series
   (sm120) GPU will not load on Ada or Ampere. Rebuild rather than copy.
4. **Never bypass a live StreamDiff node** (`enabled=false`). On 0.5.66 that destroyed the
   node. Use `hold=true` to pause, or save the project, destroy, and reload.
5. **Never use Sentinel's Pattern source or test cards as input.** `OP_Pattern` exists so the
   control image is an authored generator or a live camera split.
6. **Do not commit weights, ONNX, engines, or recordings.** `.gitignore` enforces this.

## Verifying a build

```
python export/export_fused_unet.py --width 896 --height 512 --out onnx/896x512/fused.onnx
python export/build_engine.py --onnx onnx/896x512/fused.onnx --width 896 --height 512 \
    --out engines/896x512/unet_controlnet_ipadapter_fp16.engine --opt-level 4 --verify
python export/install_custom_pack.py --name opticalpattern --controlnet \
    --engine engines/896x512/unet_controlnet_ipadapter_fp16.engine --display "SDXL OpticalPattern CN 896x512"
```

`--verify` compares the engine against the PyTorch reference saved at export time; expect
correlation above 0.999. Then in Sentinel: set `engine_tier=1`, `engine_precision=1`,
`engine_profile=sdxl/custom/opticalpattern/896x512`, invoke `relaunch`, and confirm the log
lines `CN inputs resolved: ... type='(absent - single-type engine)'` and
`Reinitialized at 896x512 - SUCCESS`. Health alone is not proof: capture the output at
`controlnet_scale` 0 and 1.0 and confirm the control image is visibly steering the result.

Builds take longer than a 10-minute tool timeout. Run them detached (see
`export/run_lora_engines.cmd`) and watch the log files.

## Tuning facts already measured

See the tuning section of `README.md`. In short: wire the control image into Video Input as
well as Control Image, use denoise 1.0 with feedback 0 for the strongest legible figure, and
never run feedback below denoise 0.9 with this engine.

## Working with the Sentinel MCP tools

Follow the Sentinel workspace manual (`CLAUDE.md` in the Sentinel workspace) for live authoring
discipline: one node at a time, inspect real health and frames, prove with captures, and use
`capture_at` overrides rather than mutating a user's live parameters during experiments.
