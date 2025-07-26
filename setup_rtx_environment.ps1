# JustNews V4: RTX AI Toolkit Environment Setup
# This script prepares the RTX 3090 environment for V4 development

Write-Host "🚀 JustNews V4: RTX AI Toolkit Environment Setup" -ForegroundColor Cyan

# 1. Verify RTX 3090 Hardware
Write-Host "`n1. Verifying RTX 3090 Hardware..." -ForegroundColor Yellow
nvidia-smi
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ NVIDIA GPU not detected. Ensure RTX 3090 drivers are installed."
    exit 1
}

# 2. Check NVIDIA Driver Version (R535+ required)
Write-Host "`n2. Checking NVIDIA Driver Version..." -ForegroundColor Yellow
$driverVersion = nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits
Write-Host "Driver Version: $driverVersion"

# 3. Verify Docker Desktop Version (4.41+ required for GPU support)
Write-Host "`n3. Verifying Docker Desktop Version..." -ForegroundColor Yellow
docker --version
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Docker GPU support not available. Enable in Docker Desktop settings."
    exit 1
}

# 4. Python Environment Setup
Write-Host "`n4. Setting up Python Environment..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install tensorrt tensorrt-llm nvidia-tensorrt

# 5. Verify TensorRT Installation
Write-Host "`n5. Verifying TensorRT Installation..." -ForegroundColor Yellow
python -c "import tensorrt; print(f'✅ TensorRT version: {tensorrt.__version__}')"

# 6. Create V4 Development Branch
Write-Host "`n6. Creating V4 Development Branch..." -ForegroundColor Yellow
git checkout -b v4-rtx-development
git add .
git commit -m "Initialize V4 RTX AI Toolkit development environment"

Write-Host "`n🎉 RTX 3090 environment ready for V4 development!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Apply for NVIDIA AIM SDK early access" -ForegroundColor White
Write-Host "2. Install NVIDIA AI Workbench" -ForegroundColor White  
Write-Host "3. Implement RTX-optimized hybrid inference" -ForegroundColor White
