@echo off
set HF=C:\Users\times\Sentinel\opticalpattern_build\venv\Scripts\hf.exe
set W=C:\Users\times\Sentinel\opticalpattern_build\weights
"%HF%" download SG161222/RealVisXL_V5.0 --include "unet/*" --local-dir "%W%\realvisxl-v5" > "%W%\download_realvis_unet.log" 2>&1
"%HF%" download SG161222/RealVisXL_V5.0 --include "scheduler/*" --local-dir "%W%\realvisxl-v5" >> "%W%\download_realvis_unet.log" 2>&1
echo UNET_DONE >> "%W%\download_realvis_unet.log"
