# Discord post draft: OpticalPattern ControlNet running in StreamDiff

Got a third-party SDXL ControlNet running inside StreamDiff as a custom engine pack, and wrote it all up.

**What it is:** the SDXL OpticalPattern ControlNet (Civitai 161132, the "hidden optical illusion" model) fused with SDXL-Turbo and the SDXL IP-Adapter into one TensorRT engine that the 0.5.66 StreamDiff node loads through `manifest_custom.json`. A small Module (`OP_Pattern`) makes the control image live: either a three-tone percentile split of the camera (darkest 25% / free / brightest 25%, which is what the ControlNet was trained on) or spiral / rings / stripes / checker generators. Runs 896x512 at ~30 fps with the window closed, ~15 with it open, on an RTX 5090 Laptop.

**How the engine was made, without an export script from you:** dumped the shipped engine's bindings, identified the base UNet as SDXL-Turbo by output correlation, wrapped UNet + ControlNet + IP-Adapter processors in a diffusers module with your exact input order and names, ONNX opset 17, TensorRT 10.15.1 FP16 static profile. Verified TRT vs PyTorch at corr 0.99993. Repo has the contract dump, the export and build scripts, an installer, the Sentinel project with presets, and the tuning measurements. Everything is repeatable; the prebuilt engine is on Hugging Face for 50-series cards.

Repo: https://github.com/jhurlbut/sentinel-opticalpattern-controlnet
Engine + demos: https://huggingface.co/Jamesbass/sentinel-opticalpattern-controlnet

**Things worth knowing on your side:**
1. A single-type ControlNet engine only loads if it is named `unet_controlnet_union_ipadapter_fp16.engine`. With that name the loader logs `type='(absent - single-type engine)'` and works. The non-Union filename resolves through the legacy `engines/sdxl-turbo-ipadapter/` path and never loads.
2. `import_custom_pack` drops the engine in a `default/` folder, which shows up as `(0x0)` in the resolution list. Renaming the folder to `896x512` and editing the manifest fixes it; the enum refreshes on relaunch.
3. Setting `enabled=false` on a live StreamDiff node with the CN engine loaded destroyed the node (gone from the pipeline list, engines unloaded, app alive). `hold=true` is fine. Happy to file a proper bug report if useful.
4. IP-Adapter and integrated depth all work through the fused engine unchanged. The integrated depth engine logs a "device supports 82 SMs, engine requires 170" warning on this GPU but runs.

Question, if anyone knows: how does `denoise` map to the UNet timestep, and what update runs after the single step? It decides whether one-step models trained for a fixed timestep (DMD2, Hyper-SD) can be swapped in for the next version.

Demo videos below.
