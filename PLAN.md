# OpticalPattern ControlNet in Sentinel StreamDiff: Integration Plan

Status: plan only, nothing built. Written 2026-09-03 against Sentinel 0.5.66 on an RTX 5090 Laptop (sm120c, 24 GB).

## Target

- Model: SDXL ControlNet OpticalPattern v10e (Civitai model 161132, version 181335).
- Local file: `C:\Users\times\Downloads\sdxlControlnet_v10eOpticalpattern.safetensors` (fp16, about 4.89 GB). Expected SHA256 `10F8985748CE2FBB38E83A7FE603809C9C8F94BA0C9D42B3C6A6E88467623F0A`.
- License: CreativeML Open RAIL++-M with addendum. Read the addendum before bundling the engine into a distributable project.
- Author settings: weight 1.0, start 0.15, end 0.75, no preprocessor. Conditioning acts on the darkest 25% and brightest 25% of pixels; mid tones are unconditioned.

## How Sentinel consumes a ControlNet (verified from the live build)

- StreamDiff runs SDXL through TensorRT 10 (`nvinfer_10.dll`, CUDA 12.8 runtime). There is no runtime loader for safetensors, ONNX, or LoRA files.
- ControlNet is fused into the UNet engine. Shipped file: `engines/profiles/sdxl/896x512/unet_controlnet_union_ipadapter_fp16.engine` (8.4 GB) plus an FP8 sibling.
- Tensor bindings recovered from the shipped engine: inputs `sample`, `timestep`, `encoder_hidden_states`, `text_embeds`, `time_ids`, `controlnet_cond`, `controlnet_scale`, `control_type` (Union only), `ipadapter_scale`; output `out_sample`. Exact shapes, dtypes, and optimization profiles are still to be dumped (Phase 0).
- The binary also recognises a non-Union filename `unet_controlnet_ipadapter_fp16.engine` and guards for a missing `control_type` binding, so a single-purpose ControlNet is an anticipated case.
- Custom engines install through the undocumented `IMPORT_CUSTOM_PACK {category, src_path, display_name?}` command (StateTree action `/sentinel/engines/actions/import_custom_pack`). It copies one engine into `engines/profiles/<category>/custom/<name>/`, registers it in `engines/manifest_custom.json` as `local_only`, and the node's `engine_profile` parameter scans `profiles/sdxl/custom/`. `REMOVE_CUSTOM_PACK` and `PRUNE_MISSING_CUSTOM_PACKS` clean up.
- Relevant node parameters: `engine_profile`, `engine_precision`, `controlnet_enabled`, `controlnet_use_input`, `controlnet_scale`, `controlnet_type`, `controlnet_auto_depth`. No start/end step controls exist; the author's 0.15 to 0.75 window has to be approximated with scale.

## Route

Build a fused SDXL UNet + OpticalPattern ControlNet + IP-Adapter TensorRT engine that matches the shipped binding contract, import it as a custom `sdxl` pack, and build the Sentinel graph around it. Build the graph first against the shipped Union engine so the control-image Module, prompts, and presets are proven before the custom engine exists.

## Open questions to resolve before spending build time

1. Exact binding shapes, dtypes, and min/opt/max profiles of the shipped 896x512 ControlNet engine.
2. Which base UNet Sentinel's engines were exported from (SDXL 1.0 base, or a Turbo/Lightning distill; the binary references an `sdxl-turbo-ipadapter` profile).
3. Which IP-Adapter variant and how it is fused (the `ipadapter_scale` binding and the separate `image_projection_fp16.engine` suggest image tokens are appended to `encoder_hidden_states`).
4. The exact TensorRT minor version used to build the shipped engines. A plan deserialises only on the same TensorRT major.minor, so the build toolchain must match `nvinfer_10.dll`.
5. Whether OODLabs will share their export script. That collapses Phases 2 and 3 to a config change. Ask first.

## Phases

### Phase 0: Contract discovery (about half a day)

1. Create a Python 3.12 venv with `tensorrt` matching the shipped `nvinfer_10.dll`, plus `polygraphy` and `onnx`.
2. Dump every binding name, dtype, shape, and optimization profile from `unet_controlnet_union_ipadapter_fp16.engine` and `unet_ipadapter_fp16.engine`. Save the table to `projects/opticalpattern_controlnet/contract/bindings.json`.
3. Import a copy of the shipped Union engine through `IMPORT_CUSTOM_PACK` with a throwaway display name. Confirm it appears in `engine_status` as `local_only`, that `engine_profile` lists it on a StreamDiff node, and that the node goes healthy. Remove it with `REMOVE_CUSTOM_PACK delete_files=true`. This proves the import path with zero build risk.
4. Email OODLabs for the exporter, base checkpoint, and IP-Adapter variant.

Exit: bindings table saved, custom import proven end to end, base model identified or a decision made to proceed with SDXL 1.0 base and verify visually.

### Phase 1: Sentinel graph against the shipped Union engine (1 day, can overlap Phase 0)

1. Author `OP_Pattern`, a Module that turns a meaningful source (camera, video, or an authored generator) into the three-tone control image: darkest 25% to black, brightest 25% to white, everything else to mid grey, with percentile thresholds measured from the live histogram and a coverage control. The preview must show the three-tone result.
2. Create `streamdiff` at 896x512 on the Union FP16 engine, wire the source to the main input and `OP_Pattern` to the Control Image, set `controlnet_use_input=false`, and prove health, climbing frames, and a non-blank capture.
3. Save narrow presets: a `controlnet_scale` ladder (0.4 / 0.6 / 0.8 / 1.0) and a prompt bank for illusion subjects.
4. Capture a baseline sweep so the custom engine has something to be compared against.

Exit: graph healthy, control image legible, baseline captures saved.

### Phase 2: Environment and weights (half a day, mostly downloads)

1. Same venv or a second one: `torch` for CUDA 12.8, `diffusers`, `transformers`, `accelerate`, `safetensors`, `onnx`, `onnx-graphsurgeon`, `onnxruntime-gpu`.
2. Verify the SHA256 of the downloaded OpticalPattern file and copy it to `projects/opticalpattern_controlnet/weights/` (this folder stays out of any public repo).
3. Download the base UNet identified in Phase 0 and the IP-Adapter SDXL weights that match the shipped image encoder.
4. Disk: allow 60 GB for weights, ONNX with external data, and TensorRT timing caches. 1.5 TB is free.

Exit: all weights on disk with verified hashes.

### Phase 3: Fused ONNX export (1 to 2 days)

1. Write `export/fused_unet.py`: one `nn.Module` wrapping `UNet2DConditionModel` with IP-Adapter attention processors and `ControlNetModel` loaded from the OpticalPattern file. Forward signature must be exactly the shipped binding order minus `control_type`: `(sample, timestep, encoder_hidden_states, text_embeds, time_ids, controlnet_cond, controlnet_scale, ipadapter_scale) -> out_sample`.
2. Export with `torch.onnx.export`, opset 17 or 18, fp16, external data, dynamic axes matching the shipped profiles.
3. Validate with onnxruntime against the PyTorch forward on a fixed latent; require max abs error under 1e-2 in fp16.
4. Run `polygraphy surgeon sanitize --fold-constants` and confirm the graph still has the same inputs and outputs.

Exit: sanitized ONNX that reproduces PyTorch output and carries the shipped binding names.

### Phase 4: TensorRT build (2 to 4 hours wall time per precision)

1. `trtexec --onnx=fused.onnx --fp16 --saveEngine=unet_controlnet_ipadapter_fp16.engine` with `--minShapes/--optShapes/--maxShapes` copied from the Phase 0 profile dump, `--builderOptimizationLevel=4`, and a persistent `--timingCacheFile`.
2. Build 896x512 first. Build 512x896 only after 896x512 is proven in the app.
3. Smoke test with `polygraphy run --trt` and compare against onnxruntime on the same inputs.
4. FP8 is optional and later. It needs calibration or ModelOpt quantisation and is not required for a first result.

Exit: an engine file that deserialises under the shipped TensorRT and matches onnxruntime within tolerance.

### Phase 5: Import and load in Sentinel (half a day)

1. `IMPORT_CUSTOM_PACK category=sdxl src_path=<engine> display_name="SDXL OpticalPattern CN 896x512"`.
2. `engine_status` must list the new pack as `complete` and `local_only=true`.
3. On the Phase 1 StreamDiff node, set `engine_profile` to the custom pack, `engine_precision=FP16`, `controlnet_enabled=true`. Wait for reinitialisation, then check `stats.healthy`, `statusMessage`, `framesProcessed`, and `sentinel.log` for binding or shape errors.
4. If the loader rejects the non-Union file, fall back to exporting with a dummy `control_type` input and the Union filename, and record which path worked.

Exit: node healthy on the custom engine with a non-blank, pattern-driven capture.

### Phase 6: Tune, prove, bundle (1 day)

1. Sweep `controlnet_scale` with `capture_at` and `sweep_record`. The author's 0.15 to 0.75 window has no direct control here, so find the scale where the illusion reads without the pattern dominating. Expect a lower scale than 1.0 because StreamDiffusion applies the ControlNet on every denoise step.
2. Run `sentinel_vision eval_pipeline preset=render_quality` on the best captures and compare against the Phase 1 Union baseline.
3. Profile `cook_hz` on the custom engine versus Union and publish the ratio in the README.
4. Bake tuned defaults, save the project with bundled modules under `projects/opticalpattern_controlnet/`, write the README with the export recipe, hashes, license notes, and what was not verified.

Exit: saved project, proof bundle, README.

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| TensorRT minor mismatch | Engine will not deserialise | Pin the venv `tensorrt` to the shipped DLL version in Phase 0 before any build |
| Wrong base UNet | Engine loads but the look drifts from the shipped profiles | Identify the base in Phase 0; compare a no-ControlNet capture against the shipped engine |
| IP-Adapter fusion differs | Shape error on `encoder_hidden_states` or `ipadapter_scale` | Dump the shipped profile shapes and mirror them exactly; ask OODLabs |
| Loader rejects non-Union file | Custom engine never selected | Phase 5 fallback: dummy `control_type` input and Union filename |
| VRAM | Fused fp16 engine near 10 GB plus 5 GB core leaves little for depth or matting nodes | Keep other engine nodes off during proof; FP8 build later if needed |
| Few-step schedule weakens the illusion | Pattern either invisible or overpowering | Scale ladder presets, prompt bank, three-tone coverage control in `OP_Pattern` |
| License addendum | Distribution limits on the bundled engine | Read before bundling; keep the engine and weights out of any public project repo |

## Effort

Roughly five to seven working days if OODLabs does not share their exporter, two to three if they do. Phases 0 and 1 run in parallel and are the only work that should start before the open questions are answered.
