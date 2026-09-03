---
license: other
license_name: stability-ai-non-commercial-research-community
license_link: https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md
base_model:
  - stabilityai/sdxl-turbo
  - h94/IP-Adapter
tags:
  - tensorrt
  - sentinel
  - streamdiffusion
  - controlnet
  - sdxl-turbo
  - real-time
  - optical-illusion
pipeline_tag: image-to-image
---

# OpticalPattern ControlNet engine for Sentinel StreamDiff

Prebuilt TensorRT engine that fuses the SDXL-Turbo UNet, the **OpticalPattern** ControlNet
([Civitai 161132](https://civitai.com/models/161132), v10e by nacholmo) and the SDXL
IP-Adapter into the single UNet engine that [Sentinel](https://sentinel.oodlabs.com)'s
StreamDiff node loads. It makes real-time optical illusions from a live camera at 896x512.

Code, the Sentinel project, the OP_Pattern control-image Module, build scripts and tuning
notes: **https://github.com/jhurlbut/sentinel-opticalpattern-controlnet**

## Files

| File | Size | What |
|---|---|---|
| `profiles/sdxl/custom/opticalpattern/896x512/unet_controlnet_union_ipadapter_fp16.engine` | 8.4 GB | The fused engine. Must keep this filename and folder layout inside Sentinel's `engines/`. |
| `demos/op_diffusion_cloud_face.mp4` | 58 MB | Node output: a face illusion hidden in clouds |
| `demos/sentinel_window_spiral_interchange.mp4` | 45 MB | Sentinel window: highway interchange bent into a spiral |

## Compatibility

This is a TensorRT **plan**, not portable weights. It only loads on the GPU architecture it
was built for:

- NVIDIA RTX 50-series (Blackwell, sm120). Built on an RTX 5090 Laptop GPU.
- TensorRT 10.15.1 runtime, which is what Sentinel 0.5.66 ships as `nvinfer_10.dll`.
- Sentinel 0.5.66 or newer, `engine_tier` = ControlNet + IP-Adapter, FP16, 896x512.
- About 9 GB of VRAM for this engine plus Sentinel's CLIP, VAE and IP-Adapter engines.

Any other GPU generation or TensorRT version: rebuild with the scripts in the GitHub repo.
The build takes 30 to 60 minutes.

## Install

```
git clone https://github.com/jhurlbut/sentinel-opticalpattern-controlnet
python sentinel-opticalpattern-controlnet/export/install_custom_pack.py --name opticalpattern --controlnet --engine <downloaded .engine> --display "SDXL OpticalPattern CN 896x512"
```

Then open the Sentinel project from the repo, set the StreamDiff node's engine resolution to
`opticalpattern (896x512)` and relaunch. Full steps and the measured tuning recipe are in the
repo README.

## Binding contract

Static batch 1. Inputs `sample` [1,4,64,112] f16, `timestep` [1] f32, `encoder_hidden_states`
[1,81,2048] f16 (77 text + 4 IP-Adapter tokens), `text_embeds` [1,1280] f16, `time_ids`
[1,6] f32, `controlnet_cond` [1,3,512,896] f16, `controlnet_scale` [1] f32,
`ipadapter_scale` [70] f32. Output `out_sample` [1,4,64,112] f16. Verified against the
PyTorch reference at correlation 0.99993.

## License

The engine is a derivative of SDXL-Turbo (Stability AI Non-Commercial Research Community
License), the OpticalPattern ControlNet (CreativeML Open RAIL++-M with the author's addendum)
and IP-Adapter (Apache-2.0). It is provided for non-commercial research use only; commercial
use needs a Stability AI license. The ControlNet training weights themselves are not
redistributed here.
