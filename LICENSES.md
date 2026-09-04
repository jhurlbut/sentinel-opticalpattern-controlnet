# Licenses and redistribution

The code in this repository (scripts, Module HLSL, manifests, docs) is MIT licensed; see
`LICENSE`.

The engine is a derivative of several models with their own terms. Anyone redistributing or
using a built engine is bound by all of them:

| Component | Source | License |
|---|---|---|
| SDXL-Turbo UNet | stabilityai/sdxl-turbo | Stability AI Non-Commercial Research Community License |
| OpticalPattern ControlNet v10e | Civitai model 161132 (nacholmo) | CreativeML Open RAIL++-M with author addendum |
| IP-Adapter SDXL | h94/IP-Adapter | Apache-2.0 |
| ControlNet config | diffusers/controlnet-canny-sdxl-1.0 | OpenRAIL++ |
| LoRAs used by `run_lora_engines.cmd` | Civitai, per model page | Per model page |
| RealVisXL V5.0 UNet (realvis pack) | SG161222/RealVisXL_V5.0 | CreativeML Open RAIL++-M |
| DMD2 SDXL 4-step LoRA (realvis pack) | tianweiy/DMD2 | Apache-2.0, research-use note on the SDXL weights |

The prebuilt engine published alongside this repo is therefore for non-commercial research
use only, and only runs on the GPU architecture it was built for. Commercial use requires a
Stability AI license for SDXL-Turbo. The ControlNet weights are not included here; download
them from the Civitai page and accept its terms.
