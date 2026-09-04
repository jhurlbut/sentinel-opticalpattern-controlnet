@echo off
setlocal
set B=C:\Users\times\Sentinel\opticalpattern_build
set PY=%B%\venv\Scripts\python.exe
set EXP=C:\Users\times\Sentinel\workspace\projects\opticalpattern_controlnet\export
cd /d %B%
set NAME=%1
if "%NAME%"=="" set NAME=realvis_896x512
if not exist calib mkdir calib
if not exist calib\%NAME%.npz (
  echo === %NAME% calib ===
  %PY% -u %EXP%\make_calib.py --out calib\%NAME%.npz --n 32 > logs\calib_%NAME%.log 2>&1
  if errorlevel 1 ( echo %NAME% CALIB_FAILED & exit /b 1 )
)
if not exist onnx\%NAME%_fp8\fused_fp8.onnx (
  echo === %NAME% quantize ===
  %PY% -u %EXP%\quantize_fp8.py --onnx onnx\%NAME%\fused.onnx --calib calib\%NAME%.npz --out onnx\%NAME%_fp8\fused_fp8.onnx > logs\quantize_%NAME%.log 2>&1
  if errorlevel 1 ( echo %NAME% QUANTIZE_FAILED & exit /b 1 )
)
echo === %NAME% fp8 build ===
if not exist engines\%NAME%_fp8 mkdir engines\%NAME%_fp8
%PY% -u %EXP%\build_engine.py --fp8 --onnx onnx\%NAME%_fp8\fused_fp8.onnx --width 896 --height 512 --out engines\%NAME%_fp8\unet_controlnet_union_ipadapter_fp8.engine --opt-level 4 --timing-cache fp8_timing.cache --verify > logs\build_%NAME%_fp8.log 2>&1
if errorlevel 1 ( echo %NAME% FP8_BUILD_FAILED & exit /b 1 )
echo %NAME% FP8_DONE
exit /b 0
