@echo off
rem Upload the prebuilt engine, demo recordings and model card to Hugging Face.
rem Requires a one-time login first:  venv\Scripts\hf.exe auth login
setlocal
set HF=C:\Users\times\Sentinel\opticalpattern_build\venv\Scripts\hf.exe
set REPO=jhurlbut/sentinel-opticalpattern-controlnet
set ENG=C:\Users\times\Sentinel\engines\profiles\sdxl\custom\opticalpattern\896x512\unet_controlnet_union_ipadapter_fp16.engine
set PRJ=C:\Users\times\Sentinel\workspace\projects\opticalpattern_controlnet
%HF% repo create %REPO% --repo-type model 2>nul
%HF% upload %REPO% "%PRJ%\docs\hf\README.md" README.md --repo-type model || exit /b 1
%HF% upload %REPO% "%PRJ%\captures\videos\OP_Diffusion_20260903_155351.mp4" demos/op_diffusion_cloud_face.mp4 --repo-type model || exit /b 1
%HF% upload %REPO% "%PRJ%\captures\videos\Sentinel_Window_20260903_160327.mp4" demos/sentinel_window_spiral_interchange.mp4 --repo-type model || exit /b 1
%HF% upload %REPO% "%ENG%" profiles/sdxl/custom/opticalpattern/896x512/unet_controlnet_union_ipadapter_fp16.engine --repo-type model || exit /b 1
echo HF_UPLOAD_DONE
