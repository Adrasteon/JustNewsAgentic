@echo off
echo Launching JustNews V4 GPU Acceleration in WSL
echo Target: 42.1 articles/sec processing!

echo Starting WSL with GPU environment...
wsl bash -c "cd /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment && bash start_gpu_analyst.sh"

pause
