@echo off
setlocal
set B=C:\Users\times\Sentinel\opticalpattern_build
set PY=%B%\venv\Scripts\python.exe
set EXP=C:\Users\times\Sentinel\workspace\projects\opticalpattern_controlnet\export
cd /d %B%
call :one 1024x768 1024 768
call :one 1024x576 1024 576
echo PROFILES_ALL_DONE
exit /b 0
:one
set NAME=%1
set W=%2
set H=%3
if exist engines\%NAME%\unet_controlnet_ipadapter_fp16.engine ( echo %NAME% already built & exit /b 0 )
if exist onnx\%NAME%\fused.onnx.ref.pt ( echo === %NAME% export skipped === & goto build )
echo === %NAME% export ===
if not exist onnx\%NAME% mkdir onnx\%NAME%
%PY% -u %EXP%\export_fused_unet.py --width %W% --height %H% --out onnx\%NAME%\fused.onnx > logs\export_%NAME%.log 2>&1
if errorlevel 1 ( echo %NAME% EXPORT_FAILED & exit /b 0 )
:build
echo === %NAME% build ===
if not exist engines\%NAME% mkdir engines\%NAME%
%PY% -u %EXP%\build_engine.py --onnx onnx\%NAME%\fused.onnx --width %W% --height %H% --out engines\%NAME%\unet_controlnet_ipadapter_fp16.engine --opt-level 4 --verify > logs\build_%NAME%.log 2>&1
if errorlevel 1 ( echo %NAME% BUILD_FAILED & exit /b 0 )
echo %NAME% DONE
exit /b 0
