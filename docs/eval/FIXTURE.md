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
