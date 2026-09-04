# Plan v2: a higher-detail OpticalPattern engine for Sentinel

Status: in progress 2026-09-03. Phase 0 and Phase 1 done (see `docs/eval/FIXTURE.md`).
Phase 2 finding: the StreamDiff node caps `processing_width` and `processing_height` at 1024,
so the 1280x768 engine built and verified but cannot be loaded (`Failed to reload engines ...
at 1024x768`). Replacement profiles: 1024x768 (0.79 MP) and 1024x576 (0.59 MP, 16:9).

## Where the detail is lost today

The shipped v1 engine is SDXL-Turbo, one denoising step, 896x512, FP16. Measured on the
RTX 5090 Laptop: 34 ms UNet step, about 15 fps with the node window open, 30 fps closed.
Everything in it works, but the images are soft. Four causes, in order of weight:

1. **One step from pure noise.** Turbo at denoise 1.0 resolves the large tonal shapes the
   ControlNet asks for, not surface detail. Lower denoise needs a good init, and the only
   inits available (the pattern, or the feedback frame) either take over or run away.
2. **Base model.** SDXL-Turbo is a 512-native distillation of SDXL base. The Civitai author's
   examples are RealVisXL 2.0 at 20 steps, a photoreal fine-tune at 1024.
3. **Resolution.** 896x512 is 0.46 megapixels. The author works at 1024x1024.
4. **No negative prompt, no CFG.** Sentinel runs at guidance 1. Nothing can be done about
   this inside the engine; it is a property of the node.

Cause 4 is fixed. Causes 1 to 3 each have a lever, and they stack.

## Success criteria

Judge every candidate on the same fixture: the saved project, the spiral control image at
coverage 0.25, the forest prompt and the "cinematic scenery city street" prompt, seed 42
locked, ControlNet scale 0.8 and 1.0, captured with `capture_at`.

| Metric | v1 baseline | Target |
|---|---|---|
| Laplacian-variance sharpness of the 896x512 output | to be measured on v1 first | at least 2x v1 |
| Figure legibility (spiral visible at scale 0.8 in a 448-px thumbnail) | yes | yes, unchanged |
| Cracked or fragmented surfaces | none at denoise 1.0 | none |
| Frame rate, node window closed | about 30 fps | at least 12 fps for the "Live" rung, any rate for a "Beauty" rung |
| VRAM | about 12 GB total | under 20 GB total so the LoRA and depth engines still fit |

Ship the result as a quality ladder of engine profiles, not one engine: Live (fast), Beauty
(slow, capture-grade), with the project defaulting to Live.

## Phase 0: measure and de-risk (no builds, half a day)

- Add a sharpness script (`docs/eval/sharpness.py`, Laplacian variance plus a 4x downscale
  ratio) and run it on the v1 fixture captures so every later number has a baseline.
- Establish how Sentinel maps `denoise` to the UNet `timestep` and what scheduler update it
  applies after the single step. Method: feed the engine known inputs through the node at
  denoise 1.0, 0.75 and 0.5 while logging, or ask OODLabs in the Discord thread. This decides
  whether a UNet trained for a fixed single timestep (DMD2 at 399, Hyper-SD at 800) can be
  driven at all. Assumption to verify: `timestep = round(999 * denoise)` and an epsilon-
  prediction Euler update to x0.
- Confirm two StreamDiff nodes pointing at the same profile share one engine in the
  `SharedEnginePool` (the log prints `refs=N`). This gates Phase 1.

## Phase 1: two-step sampling with two nodes (no new engine, one day)

Turbo is trained for 1 to 4 steps, and two steps are dramatically sharper than one. Sentinel
exposes one step per node, so chain two nodes:

```
OP_Pattern.Control -> OP_Diffusion  (denoise 1.0, cn 0.8..1.0)   -> OP_Refine (denoise 0.35..0.5, cn 0.6..0.8)
OP_Pattern.Control -> OP_Refine.Control Image
```

The first node lays out the illusion, the second re-denoises its output with the same
control image, which is exactly a second Turbo step. Expected cost: two UNet steps per frame,
so half the frame rate; VRAM roughly unchanged if the pool shares the engine.

Prove: sharpness metric on the refine output versus the v1 output, and that the figure survives
the refine at scale 0.6 to 0.8. Save `Refine Off / Light / Full` presets on OP_Refine scoped
to `denoise, controlnet_scale, feedback`. If the pool does not share, this costs 8 GB and is
only viable as the Beauty rung.

Also in this phase, and independent of it: put a `vsr` (upscaler) node after the last
StreamDiff node. The author recommends 4x-UltraSharp for the same softness; Sentinel's VSR is
the built-in equivalent. Measure whether 896x512 -> 1792x1024 VSR reads sharper than a native
higher-resolution engine would, because it is nearly free.

## Phase 2: a 1024-class profile of the current recipe (one build, two to three hours)

Rebuild the v1 export at a higher static resolution. Candidates, all multiples of 64:

| Profile | Megapixels | Expected UNet step | Note |
|---|---|---|---|
| 896x512 (v1) | 0.46 | 34 ms | baseline |
| 1152x640 | 0.74 | about 55 ms | keeps 16:9-ish framing |
| 1280x768 | 0.98 | about 75 ms | SDXL-native pixel count, 5:3 |
| 1024x1024 | 1.05 | about 80 ms | the author's format |

Sentinel accepts custom profile folders named `<WxH>`, so each is a new pack under
`profiles/sdxl/custom/opticalpattern/<WxH>/`. Build 1280x768 first. Turbo was distilled at
512 and can drift at 1024; the ControlNet's strong layout prior helps here, but check for
doubled figures at scale 0.6.

Prove: sharpness metric at matched output size, frame rate, and that the depth and IP-Adapter
paths still initialise (they resize internally, but confirm in the log).

## Phase 3: a better base model (the real upgrade, two to four days)

Replace SDXL-Turbo with a base that is both photoreal and good at one or two steps. The
export script already separates base UNet, ControlNet and IP-Adapter, so this is a change of
`TURBO_UNET` plus a LoRA fuse, not a rewrite. Ranked candidates:

1. **RealVisXL V5.0 + DMD2 4-step LoRA, fused.** Directly matches the author's checkpoint
   family. DMD2 is epsilon-prediction and works with the standard schedulers at 1 to 4
   steps, so it is the most likely to drop into Sentinel's fixed sampler. Fuse the LoRA into
   the UNet at export, exactly as `fuse_lora` already does for style LoRAs.
2. **DMD2 SDXL 1-step UNet** (`tianweiy/DMD2`). Best pure one-step quality published for
   SDXL, but it is a full UNet trained for timestep 399; needs the Phase 0 timestep answer.
3. **Hyper-SD SDXL 1-step / 2-step**. Strong, but relies on the TCD scheduler; riskier fit.
4. **SDXL-Lightning 1-step**. Excluded: its 1-step UNet is x0-prediction and will not match
   an epsilon-prediction sampler. Its 2- and 4-step LoRAs are epsilon-prediction and remain
   a fallback for the two-node chain from Phase 1.

Gate before any TensorRT build: run the candidate in plain PyTorch with the exact inputs
Sentinel sends (one step, the Phase 0 timestep, the OpticalPattern residuals, IP-Adapter at
scale 0) and confirm it decodes to a clean image. A candidate that needs a different
scheduler fails here, cheaply, instead of after a 40-minute build.

Prove: the full fixture, sharpness metric, and a side-by-side with the Civitai samples.
Publish the winner as a second pack (`realvis-dmd2` or similar) rather than replacing v1,
because Turbo's temporal behaviour with feedback may still be preferable for some looks.

## Phase 4: speed to pay for the quality (one to two days)

Everything above costs frame rate. Two levers buy it back:

- **FP8 build.** Sentinel exposes `engine_precision = FP8` for Ada and Blackwell. Build the
  Phase 3 winner with TensorRT FP8 quantisation (needs a calibration pass; ModelOpt or the
  TensorRT quantisation toolkit). Expect roughly 1.6x on the UNet step. Verify that the
  ControlNet residual path survives quantisation: compare against the FP16 engine on the
  fixture, not just against PyTorch.
- **Analysis proxy for OP_Pattern.** Unchanged cost today, but at 1280x768 the histogram
  sample grid should stay at 224x128; it already does. Nothing to do unless profiling says so.

## Deliverables

- New packs: `opticalpattern/1280x768`, `<newbase>/896x512`, `<newbase>/1280x768`, and an FP8
  variant of the best one, each installed through `install_custom_pack.py` and uploaded to the
  same Hugging Face repo under their profile paths.
- Project update: `OP_Refine` node and `VSR` node with narrow presets, engine quality ladder
  presets on `OP_Diffusion` (`Live`, `Beauty`), README quality table with the measured
  sharpness ratios and frame rates for every rung, stating what else was running.
- `docs/eval/` with the fixture definition, the sharpness script, and the comparison sheets.

## Risks

- Sentinel's sampler is a black box. If `denoise` does not map cleanly to a timestep, every
  fixed-timestep candidate in Phase 3 is out and only LoRA-on-standard-UNet candidates remain.
- VRAM. Two 8 GB engines plus CLIP, VAE, depth and IP-Adapter is about 21 GB if the pool does
  not share. The Beauty rung may have to be capture-only on this machine.
- Licensing widens: RealVisXL is CreativeML OpenRAIL++-M, DMD2 is Apache-2.0 with a
  research-use note on its SDXL weights. Update `LICENSES.md` per pack.
- Time. Each TensorRT build is 30 to 60 minutes and the machine is unusable for Sentinel
  while it runs, so builds go overnight or between sessions.
