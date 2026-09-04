@echo off
setlocal
set B=C:\Users\times\Sentinel\opticalpattern_build
set PY=%B%\venv\Scripts\python.exe
set EXP=C:\Users\times\Sentinel\workspace\projects\opticalpattern_controlnet\export
cd /d %B%
set NAME=realvis_896x512
if exist engines\%NAME%\unet_controlnet_ipadapter_fp16.engine ( echo %NAME% already built & goto done )
if exist onnx\%NAME%\fused.onnx.ref.pt ( echo === %NAME% export skipped === & goto build )
echo === %NAME% export ===
if not exist onnx\%NAME% mkdir onnx\%NAME%
%PY% -u %EXP%\export_fused_unet.py --base-unet weights\realvisxl-v5\unet --variant fp16 --lora weights\dmd2\dmd2_sdxl_4step_lora_fp16.safetensors --lora-scale 1.0 --width 896 --height 512 --out onnx\%NAME%\fused.onnx > logs\export_%NAME%.log 2>&1
if errorlevel 1 ( echo %NAME% EXPORT_FAILED & exit /b 1 )
:build
echo === %NAME% build ===
if not exist engines\%NAME% mkdir engines\%NAME%
%PY% -u %EXP%\build_engine.py --onnx onnx\%NAME%\fused.onnx --width 896 --height 512 --out engines\%NAME%\unet_controlnet_ipadapter_fp16.engine --opt-level 4 --verify > logs\build_%NAME%.log 2>&1
if errorlevel 1 ( echo %NAME% BUILD_FAILED & exit /b 1 )
echo %NAME% DONE
:done
exit /b 0
