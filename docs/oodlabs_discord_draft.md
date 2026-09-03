# Discord message draft for OODLabs

> Written before the base UNet was identified. It is SDXL-Turbo (see `contract/CONTRACT.md`); question 1 is answered.

Hi OODLabs team. I'm building a custom StreamDiff engine for Sentinel 0.5.66 (RTX 5090 Laptop, sm120c) so I can run the SDXL OpticalPattern ControlNet (Civitai 161132) inside StreamDiff, and I'd like to confirm a few things about the export recipe before I burn a lot of build time.

What I've already worked out from the shipped engines and `IMPORT_CUSTOM_PACK`:
- TensorRT 10.15.1, static batch-1 profiles, `sample` [1,4,64,112] for 896x512.
- Fused UNet + ControlNet + IP-Adapter in one engine. Bindings: `sample`, `timestep`, `encoder_hidden_states` [1,81,2048], `text_embeds`, `time_ids`, `controlnet_cond` [1,3,512,896], `controlnet_scale` [1], `control_type` [1,6] (Union only), `ipadapter_scale` [70], output `out_sample`.
- IP-Adapter looks like `ip-adapter_sdxl` (ViT-bigG, 4 tokens appended after the 77 text tokens, per-attn2-layer scale).
- The custom pack importer, `manifest_custom.json` schema v2, and the `profiles/sdxl/custom/<name>/<WxH>/` layout all work. The importer's own `default` folder gives an `0x0` resolution label, but renaming the folder to `896x512` resolves it.

Questions:
1. Which base UNet were the shipped SDXL engines exported from? SDXL 1.0 base does not match the engine output (corr 0.95, not fp16 noise), so I assume a distilled variant (Turbo / Lightning / LCM). Which one, and which checkpoint/revision?
2. How is `ipadapter_scale` [70] applied inside the graph: one scale per attn2 processor in diffusers module order? And is it applied to the IP branch only (`hidden = text_attn + scale * ip_attn`)?
3. For the non-Union engine the loader mentions (`unet_controlnet_ipadapter_fp16.engine`): is dropping `control_type` the only difference, and is `controlnet_scale` multiplied onto the residuals inside the graph as in diffusers?
4. Timestep convention: is `timestep` the raw scheduler timestep in [0, 999] as a float, and does the runtime ever send a batch other than 1?
5. `time_ids` order: original size, crop, target size as in diffusers (`[h, w, 0, 0, h, w]`)?
6. Any chance you can share the ONNX export / trtexec script you use for the SDXL profiles? Even the diffusers wrapper's forward signature would save a lot of guesswork.
7. Is `build_metadata.trt_version` / `base_model` in `manifest_custom.json` read by the app anywhere, or purely informational?

Happy to share the resulting engine build recipe back once it works, and to file whatever you learn from this as docs for the custom pack feature.
