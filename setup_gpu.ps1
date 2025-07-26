# JustNewsAgentic GPU Setup Script for Windows
# This script helps set up the environment for GPU-accelerated inference

Write-Host "🚀 JustNewsAgentic GPU Setup" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green

# Check for NVIDIA GPU
try {
    $gpu = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    if ($gpu) {
        Write-Host "✅ NVIDIA GPU detected: $($gpu.Name)" -ForegroundColor Green
    } else {
        Write-Host "⚠️  No NVIDIA GPU detected. CPU inference will be used." -ForegroundColor Yellow
    }
} catch {
    Write-Host "❓ Could not detect GPU information" -ForegroundColor Yellow
}

# Check for Docker
try {
    $dockerVersion = docker --version 2>$null
    if ($dockerVersion) {
        Write-Host "✅ Docker detected: $dockerVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ Docker not found. Please install Docker Desktop." -ForegroundColor Red
        Write-Host "   Download from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Docker not found. Please install Docker Desktop." -ForegroundColor Red
}

# Check for NVIDIA Container Toolkit
try {
    $nvmlResult = docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ NVIDIA Container Toolkit is working" -ForegroundColor Green
    } else {
        Write-Host "❌ NVIDIA Container Toolkit not working" -ForegroundColor Red
        Write-Host "   Install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Could not test NVIDIA Container Toolkit" -ForegroundColor Red
}

# Setup environment file
if (Test-Path ".env") {
    Write-Host "✅ .env file already exists" -ForegroundColor Green
} elseif (Test-Path ".env.example") {
    Copy-Item ".env.example" ".env"
    Write-Host "✅ Created .env file from template" -ForegroundColor Green
    Write-Host "   ⚠️  Please edit .env and add your HUGGINGFACE_TOKEN" -ForegroundColor Yellow
} else {
    Write-Host "❌ .env.example not found" -ForegroundColor Red
}

Write-Host "`n📝 Next steps:" -ForegroundColor Cyan
Write-Host "1. Get your Hugging Face token from: https://huggingface.co/settings/tokens"
Write-Host "2. Request access to Mistral models: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3"
Write-Host "3. Edit .env file and add your HUGGINGFACE_TOKEN"
Write-Host "4. Run: docker-compose up --build analyst"

Write-Host "`n🎉 Setup complete!" -ForegroundColor Green
