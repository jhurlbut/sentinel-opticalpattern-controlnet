# Quality fixture

Every engine variant in Plan v2 is judged on the same captures so the numbers compare.

| Item | Value |
|---|---|
| Project | `opticalpattern_controlnet.sentinel` as committed |
| Control image | OP_Pattern, pattern Spiral, coverage 0.25, scale 1.5, arms 3, twist 1.192, spin 0, softness 0.02 |
| OP_Pattern resolution | matches the engine profile (896x512 or 1280x768) |
| Prompt A | `dense autumn forest, winding dirt path, golden afternoon light, oil painting, highly detailed` |
| Prompt B | `cinematic scenery city street, photograph, golden hour, sharp focus` |
| Seed | 42, locked, seed travel off |
| denoise / feedback / sharpen | 1.0 / 0 / 0.1 |
| IP-Adapter | disabled |
| controlnet_scale | 0.8 and 1.0 |
| Capture | `sentinel_capture capture_at`, settle 8 frames, `max_width` left at native so the metric script does the resize |
| Metric | `python docs/eval/sharpness.py --width 896 --baseline <v1 capture> <candidates>` |
| Frame rate | `sentinel_graph profile`, node window closed, nothing else running; report as a ratio to v1 |

## v1 baseline (2026-09-03, Sentinel 0.5.66, RTX 5090 Laptop)

| Capture | lapvar x1e4 | hf_ratio |
|---|---|---|
| Prompt A, cn 0.8, denoise 1.0 (`op_exp3_fulldenoise_5.png`) | 4.65 | 0.016 |
| Prompt A, cn 1.0, denoise 1.0 (`op_exp3_fulldenoise_6.png`) | 2.66 | 0.018 |
| Prompt A, cn 0.8, feedback 1.0 at denoise 0.95 (`op_exp6_feedback95.png`) | 17.45 | 0.041 |

For contrast, the cracked empty-init captures score above 1200 on lapvar: noise counts as
edges, which is why hf_ratio is read alongside it and why the fixture pins denoise 1.0.

v1 UNet step: 34 ms at 896x512. 1280x768 build of the same recipe: 106 ms (TRT verify run).

## Plan v2 results (2026-09-03, same machine)

Two-node chain, forest prompt, spiral, seed 42. Baseline is the stage-1 node alone at
denoise 1.0 / cn 0.8 with the quarter-strength init (`op_exp11_stage1.png`).

| Configuration | lapvar x1e4 | hf_ratio | vs stage 1 | Figure |
|---|---|---|---|---|
| stage 1 only | 2.03 | 0.011 | 1.00x / 1.00x | strong spiral, soft |
| + refine denoise 0.3, cn 0.7 | 1.87 | 0.010 | 0.93x / 0.84x | spiral fades |
| + refine denoise 0.45, cn 0.7 | 2.26 | 0.012 | 1.12x / 1.09x | spiral faint |
| + refine denoise 0.6, cn 0.7 | 3.18 | 0.016 | 1.57x / 1.39x | spiral gone, best plain forest |
| + refine denoise 0.75, cn 0.7 | 6.36 | 0.018 | 3.14x / 1.57x | spiral gone |
| + refine denoise 0.6, cn 1.0 | 2.36 | 0.017 | 1.16x / 1.46x | spiral clear, detailed |
| + refine denoise 0.75, cn 1.0 | 3.20 | 0.016 | 1.58x / 1.41x | spiral clear, detailed |
| + refine denoise 0.6 / 0.75, cn 1.3 | 1.88 / 2.01 | 0.019 / 0.020 | ~1x / 1.7x | pattern dominant |

Chosen: refine denoise 0.7, cn 1.0 (`Refine Full` preset); `Refine Light` is 0.5 / 1.0.
Frame cost: stage 1 about 24 ms, refine about 74 ms per frame with both node windows open;
chain output about 13 fps. VRAM 16 GB for both nodes; the engine pool shares one copy
(`refs=2`) when both nodes use the same profile.

VSR (`vsr` node, 2x, quality High) after the refine: about 4 ms per frame, lapvar on a
matched crop 1.81 versus 0.49 for a Lanczos 2x of the refine output. Edge sharpening only.

Timestep probe (`export/build_timestep_probe.py`, eps~0 engine, control image as init):
init correlation with output rises 0.08 -> 0.14 -> 0.32 -> 0.62 -> 0.76 at denoise
1.0 / 0.9 / 0.75 / 0.5 / 0.25, matching a near-linear timestep = 999 * denoise with the
init blended in below 1.0. There is no single-node setting that reaches the pure-noise
timestep-750 regime the offline probe showed; a neutral init just greys the frame.

Offline one-step probes (`export/probe_candidate.py`, pure noise, cn 0.8):
SDXL-Turbo t=999 reproduces the in-app look; t=749 is sharper for both Turbo and
RealVisXL V5 + DMD2 4-step LoRA, but only reachable with a second node.

## Quality ladder (2026-09-03, RTX 5090 Laptop, all node windows open)

Same fixture. Metrics at analysis width 1024 for the 1024 rows, 896 for the 896 rows;
ratios are against the two-step 1024 result except where noted.

| Rung | Chain | lapvar x1e4 | hf_ratio | Chain frame time | Figure |
|---|---|---|---|---|---|
| Live | 896x512, 2 steps (dn1.0 cn0.8 -> dn0.7 cn1.0) | 3.20 | 0.016 | about 150 ms (13 fps) | clear |
| Beauty | 1024x768, 2 steps (same values) | 4.06 | 0.016 | about 205 ms (10 fps) | clear, real trunk detail |
| Hero | 1024x768, 3 steps (+ dn0.45 cn1.6) | 11.29 (cn1.0 measurement) | 0.110 | about 320 ms (3 fps) | clear at cn 1.6; fades at cn 1.0 |

Third-node ControlNet: 1.0 dissolves the spiral into curved trees, 1.3 keeps a partial figure,
1.6 keeps it fully while retaining the three-step detail. Third-node denoise 0.3 / 0.45 / 0.6
at cn 1.0: lapvar 7.2 / 11.3 / 13.8, hf 0.075 / 0.110 / 0.109 versus 4.06 / 0.016 for two steps.

VRAM: 17.8 GB for three nodes sharing one 1024x768 engine (`refs=3`). A freshly created
StreamDiff node first tries to load the stock 896x512 engine; with two custom-engine nodes
already resident that load fails on VRAM, the node's engine parameters reset to defaults, and
they must be set again before the relaunch onto the shared custom engine.

Precision: all engines are FP16, the format the models ship in; TRT-vs-PyTorch correlation
is 0.99998 to 0.99999, so quantisation is not where the softness comes from. FP8 would trade
quality for speed, not the reverse.

Project default is the Live rung. `Engine: OpticalPattern CN 1024x768 Beauty` on the
diffusion nodes plus `Canvas 1024x768` on OP_Pattern switch to Beauty (relaunch each node,
first node first). OP_Refine2 ships with `hold` on; clear it and recall `Refine2 Hero` for
the third step.

## Phase 3: RealVisXL V5 + DMD2 4-step LoRA base (2026-09-03, 23:20)

Engine `custom-sdxl-realvis-896x512`: RealVisXL V5.0 UNet with the DMD2 4-step LoRA fused at
1.0, OpticalPattern ControlNet, IP-Adapter, FP16, 896x512. TRT vs PyTorch corr 0.99989. In-app
UNet step 34 ms, same as Turbo. Both chain nodes must run this engine: a mixed Turbo/RealVis
chain needs two 8.4 GB engines resident, which is 24 GB on this card and hangs the device.

| Configuration (forest fixture, 896 analysis width) | lapvar x1e4 | hf_ratio | vs Turbo 2-step (3.20 / 0.016) | Figure |
|---|---|---|---|---|
| stage 1 only, cn 0.8 / 1.0, denoise 1.0 | 22.9 / 23.3 | 0.043 | 7x / 2.7x | pattern only, grainy; ControlNet overwhelms this UNet |
| stage 1 only, cn 0.15 / 0.3 / 0.5 | - | - | - | scene appears; grainy one-step image (DMD2 is a 4-step model) |
| stage 1 cn 0.3 + refine cn 0.5 dn 0.6 / 0.75 | 51.0 / 79.4 | 0.079 / 0.081 | 16x / 25x | photoreal detail, spiral subtle |
| stage 1 cn 0.3 + refine cn 1.0 dn 0.6 | 18.9 | 0.074 | 5.9x / 4.6x | illusion, detailed |
| stage 1 cn 0.3 + refine cn 1.0 dn 0.75 | 41.5 | 0.085 | 13x / 5.3x | pattern dominant |
| stage 1 cn 0.3 + refine cn 0.7 / 0.85 at dn 0.55 / 0.65 | 22.9 to 34.2 | 0.067 to 0.070 | 7x to 11x / 4.2x to 4.4x | all four: clear spiral, rich foliage and trunk detail |

Chosen: `RealVis Stage 1 (dn1 cn0.3 fb0)` and `RealVis Refine (dn0.65 cn0.85)`. Project variant:
`opticalpattern_realvis_two_pass.sentinel`. Stage one alone is not usable with this base; the
quality is entirely in the second step, which is what a distilled 4-step model is built for.
