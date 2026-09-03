@echo off
setlocal
set B=C:\Users\times\Sentinel\opticalpattern_build
set PY=%B%\venv\Scripts\python.exe
set EXP=C:\Users\times\Sentinel\workspace\projects\opticalpattern_controlnet\export
cd /d %B%

rem Each job: <name> <lora file> <scale>
call :one glitch   weights\loras\aether_glitch_v1.safetensors   1.0
call :one infrared weights\loras\zavy_infrared_sdxl.safetensors 1.0
if exist weights\loras\xray_dd_xl.safetensors call :one xray weights\loras\xray_dd_xl.safetensors 1.0
echo ALL_DONE
exit /b 0

:one
set NAME=%1
set LORA=%2
set SCALE=%3
if exist engines\%NAME%_896x512\unet_ipadapter_fp16.engine ( echo %NAME% already built, skipping & exit /b 0 )
if exist onnx\%NAME%_896x512\unet_ipadapter.onnx.ref.pt ( echo === %NAME% export skipped, onnx present === & goto build )
echo === %NAME% export ===
if not exist onnx\%NAME%_896x512 mkdir onnx\%NAME%_896x512
%PY% -u %EXP%\export_fused_unet.py --no-controlnet --lora %LORA% --lora-scale %SCALE% --width 896 --height 512 --out onnx\%NAME%_896x512\unet_ipadapter.onnx > logs\export_%NAME%.log 2>&1
if errorlevel 1 ( echo %NAME% EXPORT_FAILED & exit /b 0 )
:build
echo === %NAME% build ===
if not exist engines\%NAME%_896x512 mkdir engines\%NAME%_896x512
%PY% -u %EXP%\build_engine.py --onnx onnx\%NAME%_896x512\unet_ipadapter.onnx --width 896 --height 512 --out engines\%NAME%_896x512\unet_ipadapter_fp16.engine --opt-level 4 --verify > logs\build_%NAME%.log 2>&1
if errorlevel 1 ( echo %NAME% BUILD_FAILED & exit /b 0 )
echo %NAME% DONE
exit /b 0
