@echo off
rem Sequential GPU build queue: 1280x768 OpticalPattern profile, then the LoRA engines.
set B=C:\Users\times\Sentinel\opticalpattern_build
call "%B%\run_1280x768.cmd" > "%B%\logs\queue_1280x768.log" 2>&1
call "%B%\run_lora_engines.cmd" > "%B%\logs\lora_batch.log" 2>&1
echo QUEUE_ALL_DONE >> "%B%\logs\queue_1280x768.log"
