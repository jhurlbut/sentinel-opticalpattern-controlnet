@echo off
set HF=C:\Users\times\Sentinel\opticalpattern_build\venv\Scripts\hf.exe
set W=C:\Users\times\Sentinel\opticalpattern_build\weights
"%HF%" download stabilityai/sdxl-turbo --include "text_encoder/*" --local-dir "%W%\sdxl-turbo" > "%W%\download_probe_deps.log" 2>&1
"%HF%" download stabilityai/sdxl-turbo --include "text_encoder_2/*" --local-dir "%W%\sdxl-turbo" >> "%W%\download_probe_deps.log" 2>&1
"%HF%" download stabilityai/sdxl-turbo --include "tokenizer/*" --local-dir "%W%\sdxl-turbo" >> "%W%\download_probe_deps.log" 2>&1
"%HF%" download stabilityai/sdxl-turbo --include "tokenizer_2/*" --local-dir "%W%\sdxl-turbo" >> "%W%\download_probe_deps.log" 2>&1
"%HF%" download madebyollin/sdxl-vae-fp16-fix --include "*.json" "diffusion_pytorch_model.safetensors" --local-dir "%W%\sdxl-vae-fp16-fix" >> "%W%\download_probe_deps.log" 2>&1
echo PROBE_DEPS_DONE >> "%W%\download_probe_deps.log"
