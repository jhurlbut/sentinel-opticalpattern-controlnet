# Sentinel StreamDiff engine contract (dumped 2026-09-03, Sentinel 0.5.66, TensorRT 10.15.1.29, sm120c)

Source of truth: `bindings.json` in this folder, produced by `dump_bindings.py` against the shipped engines.

## Fused UNet engine, 896x512 profile (`unet_controlnet_union_ipadapter_fp16.engine`)

All shapes static, batch 1, one optimization profile.

| Tensor | Mode | dtype | Shape | Notes |
|---|---|---|---|---|
| `sample` | input | fp16 | [1, 4, 64, 112] | latent H x W = 512/8 x 896/8 |
| `timestep` | input | fp32 | [1] | |
| `encoder_hidden_states` | input | fp16 | [1, 81, 2048] | 77 CLIP text tokens + 4 IP-Adapter image tokens, concatenated |
| `text_embeds` | input | fp16 | [1, 1280] | SDXL pooled text embeds (added_cond_kwargs) |
| `time_ids` | input | fp32 | [1, 6] | SDXL micro-conditioning |
| `controlnet_cond` | input | fp16 | [1, 3, 512, 896] | full-res control image, NCHW |
| `controlnet_scale` | input | fp32 | [1] | conditioning scale |
| `ipadapter_scale` | input | fp32 | [70] | one scale per cross-attention (attn2) layer in the SDXL UNet |
| `control_type` | input | fp32 | [1, 6] | ControlNet Union only. Omit for a single-purpose ControlNet |
| `out_sample` | output | fp16 | [1, 4, 64, 112] | noise prediction |

The plain IP-Adapter engine has the same tensors without `controlnet_cond`, `controlnet_scale`, `control_type`.
The 512x896 profile transposes latent to [1, 4, 112, 64] and `controlnet_cond` to [1, 3, 896, 512].

## Shared core engines

| Engine | Input | Output |
|---|---|---|
| `clip/text_encoder_fp16` | `input_ids` int64 [1,77] | `last_hidden_state` fp16 [1,77,768] |
| `clip/text_encoder_2_fp16` | `input_ids` int64 [1,77] | `last_hidden_state` fp16 [1,77,1280], `pooler_output` fp16 [1,1280] |
| `ip-adapter/clip_image_encoder_fp16` | `pixel_values` fp16 [1,3,224,224] | `image_embeds` fp16 [1,1280] |
| `ip-adapter/image_projection_fp16` | `image_embeds` fp16 [1,1280] | `image_tokens` fp16 [1,4,2048] |
| `taesdxl/vae_encoder_fp16` | `image` fp16 [1,3,H,W] dynamic | `latent` fp16 [1,4,H/8,W/8] |
| `taesdxl/vae_decoder_fp16` | `latent` fp16 [1,4,h,w] dynamic | `image` fp16 [1,3,8h,8w] |

## What this identifies

- IP-Adapter: 1280-dim image embeds and 4 projected tokens of width 2048 match `h94/IP-Adapter` `sdxl_models/ip-adapter_sdxl.safetensors` (OpenCLIP ViT-bigG-14 image encoder). Not the ViT-H variant (1024-dim embeds).
- Fusion: decoupled cross-attention inside the UNet graph. Each of the 70 attn2 layers reads text tokens [0:77] and image tokens [77:81] from the same tensor and applies its own scale from `ipadapter_scale`.
- VAE: TAESD-XL, not the full SDXL VAE.
- Base UNet: **stabilityai/sdxl-turbo** (fp16 UNet). Identified by running the shipped 512x896 `unet_ipadapter_fp16` engine and candidate PyTorch UNets on identical inputs with `ipadapter_scale` = 0 (`export/identify_base_unet.py`). Results: SDXL-Turbo corr 0.99993, rel mean error 0.0074 (fp16 noise); SDXL 1.0 base corr 0.95182, rel mean error 0.32 (not a match).

## Custom pack import (proven with a renamed copy of the shipped Union engine)

`sentinel_state invoke path=/sentinel/engines/actions/import_custom_pack args={category:"sdxl", src_path:"<engine>", display_name:"<name>"}`

- Synchronous copy; the MCP call times out after 5 s on a multi-GB file but the copy completes.
- Destination: `engines/profiles/sdxl/custom/<engine basename>/default/<engine filename>`.
- Registers `custom-sdxl-<engine basename>` in `engines/manifest_custom.json` (schema_version 2) with `is_local_only: true`, `tier: custom`, `pipelines: [streamdiff]`, `spec.model_name`, `spec.resolution: "default"`, and empty `base_model` / `build_metadata.trt_version` fields intended for user-built engines.
- Remove with `/sentinel/engines/actions/remove_custom_pack args={pack_id, delete_files:true}`.
