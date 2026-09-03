@echo off
set HF=C:\Users\times\Sentinel\opticalpattern_build\venv\Scripts\hf.exe
set W=C:\Users\times\Sentinel\opticalpattern_build\weights
"%HF%" download SG161222/RealVisXL_V5.0 --include "unet/*" "model_index.json" "scheduler/*" --local-dir "%W%\realvisxl-v5" > "%W%\download_realvis.log" 2>&1
"%HF%" download tianweiy/DMD2 dmd2_sdxl_4step_lora_fp16.safetensors --local-dir "%W%\dmd2" > "%W%\download_dmd2.log" 2>&1
echo DOWNLOADS_DONE >> "%W%\download_realvis.log"
