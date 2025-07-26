# JustNews V4 Development Task Runner
# Automates first steps for V4 RTX AI Toolkit integration

param(
    [string]$Task = "all",
    [switch]$SkipEnvironment,
    [switch]$SkipAIMSDK,
    [switch]$DryRun
)

Write-Host "🚀 JustNews V4: RTX AI Toolkit Development Setup" -ForegroundColor Cyan
Write-Host "Current Phase: Phase 1 - RTX AI Toolkit Foundation" -ForegroundColor Yellow

function Show-TaskProgress {
    param([string]$TaskName, [string]$Status = "Running")
    
    $emoji = switch ($Status) {
        "Running" { "⏳" }
        "Complete" { "✅" }
        "Failed" { "❌" }
        "Skipped" { "⏭️" }
        default { "📋" }
    }
    
    Write-Host "$emoji $TaskName" -ForegroundColor $(if ($Status -eq "Failed") { "Red" } elseif ($Status -eq "Complete") { "Green" } else { "Yellow" })
}

# Task 1: Environment Validation
if (-not $SkipEnvironment -and ($Task -eq "all" -or $Task -eq "environment")) {
    Show-TaskProgress "1. Environment Validation" "Running"
    
    try {
        # Check NVIDIA GPU
        $nvidiaSmi = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   GPU: $nvidiaSmi" -ForegroundColor Green
            Show-TaskProgress "1. Environment Validation" "Complete"
        } else {
            throw "NVIDIA GPU not detected"
        }
        
        # Check Docker
        $dockerVersion = docker --version
        Write-Host "   Docker: $dockerVersion" -ForegroundColor Green
        
        # Test GPU in Docker
        $gpuTest = docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi --query-gpu=name --format=csv,noheader
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   Docker GPU Support: ✅ Working" -ForegroundColor Green
        } else {
            Write-Warning "   Docker GPU Support: ⚠️  Not configured"
        }
        
    } catch {
        Show-TaskProgress "1. Environment Validation" "Failed"
        Write-Error "Environment validation failed: $_"
        return
    }
}

# Task 2: AIM SDK Application Status
if (-not $SkipAIMSDK -and ($Task -eq "all" -or $Task -eq "aimsdk")) {
    Show-TaskProgress "2. AIM SDK Application" "Running"
    
    $aimDocPath = "docs\AIM_SDK_Application.md"
    if (Test-Path $aimDocPath) {
        Write-Host "   AIM SDK Application document created" -ForegroundColor Green
        Write-Host "   📋 Action Required: Submit application at developer.nvidia.com/aim-sdk" -ForegroundColor Yellow
        Write-Host "   📋 Include project details from $aimDocPath" -ForegroundColor Yellow
        Show-TaskProgress "2. AIM SDK Application" "Complete"
    } else {
        Write-Warning "   AIM SDK Application document not found"
        Show-TaskProgress "2. AIM SDK Application" "Failed"
    }
}

# Task 3: V4 Infrastructure Setup
if ($Task -eq "all" -or $Task -eq "infrastructure") {
    Show-TaskProgress "3. V4 Infrastructure Setup" "Running"
    
    try {
        # Check V4 files
        $v4Files = @(
            "docker-compose.v4.yml",
            "agents\analyst\Dockerfile.v4",
            "agents\analyst\requirements_v4.txt",
            "agents\analyst\rtx_manager.py"
        )
        
        $missingFiles = @()
        foreach ($file in $v4Files) {
            if (Test-Path $file) {
                Write-Host "   ✅ $file" -ForegroundColor Green
            } else {
                Write-Host "   ❌ $file (missing)" -ForegroundColor Red
                $missingFiles += $file
            }
        }
        
        if ($missingFiles.Count -eq 0) {
            Show-TaskProgress "3. V4 Infrastructure Setup" "Complete"
        } else {
            Show-TaskProgress "3. V4 Infrastructure Setup" "Failed"
            Write-Error "Missing files: $($missingFiles -join ', ')"
        }
        
    } catch {
        Show-TaskProgress "3. V4 Infrastructure Setup" "Failed"
        Write-Error "Infrastructure setup failed: $_"
    }
}

# Task 4: Development Environment Test
if ($Task -eq "all" -or $Task -eq "test") {
    Show-TaskProgress "4. Development Environment Test" "Running"
    
    if ($DryRun) {
        Write-Host "   🧪 DRY RUN - Would test V4 environment" -ForegroundColor Yellow
        Show-TaskProgress "4. Development Environment Test" "Skipped"
    } else {
        try {
            # Test Python environment
            python -c "import tensorrt; print(f'TensorRT: {tensorrt.__version__}')" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "   ✅ TensorRT available" -ForegroundColor Green
            } else {
                Write-Host "   📋 TensorRT not installed (will install with V4 requirements)" -ForegroundColor Yellow
            }
            
            # Test Docker Compose V4
            Write-Host "   🧪 Testing V4 docker-compose configuration..." -ForegroundColor Yellow
            docker-compose -f docker-compose.v4.yml config > $null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "   ✅ V4 docker-compose configuration valid" -ForegroundColor Green
                Show-TaskProgress "4. Development Environment Test" "Complete"
            } else {
                Write-Warning "   ⚠️  V4 docker-compose configuration issues detected"
                Show-TaskProgress "4. Development Environment Test" "Failed"
            }
            
        } catch {
            Show-TaskProgress "4. Development Environment Test" "Failed"
            Write-Error "Environment test failed: $_"
        }
    }
}

# Task 5: Next Steps Summary
Write-Host "`n📋 First Steps Summary:" -ForegroundColor Cyan
Write-Host "✅ Phase 1 Development Files Created:" -ForegroundColor Green
Write-Host "   • docker-compose.v4.yml - RTX-enhanced container configuration" -ForegroundColor White
Write-Host "   • agents/analyst/Dockerfile.v4 - RTX AI Toolkit container" -ForegroundColor White
Write-Host "   • agents/analyst/requirements_v4.txt - TensorRT-LLM dependencies" -ForegroundColor White
Write-Host "   • agents/analyst/rtx_manager.py - RTX optimization manager" -ForegroundColor White
Write-Host "   • setup_rtx_environment.ps1 - Environment setup automation" -ForegroundColor White

Write-Host "`n📋 Manual Actions Required:" -ForegroundColor Yellow
Write-Host "   1. 🌐 Apply for NVIDIA AIM SDK Early Access:" -ForegroundColor White
Write-Host "      https://developer.nvidia.com/aim-sdk" -ForegroundColor Cyan
Write-Host "   2. 💻 Install NVIDIA AI Workbench:" -ForegroundColor White
Write-Host "      https://developer.nvidia.com/ai-workbench" -ForegroundColor Cyan
Write-Host "   3. 🐳 Enable Docker Desktop GPU Support:" -ForegroundColor White
Write-Host "      Settings > Features in Development > Docker Model Runner" -ForegroundColor Cyan

Write-Host "`n🚀 Ready for V4 Development Phase 1!" -ForegroundColor Green
Write-Host "Next: Build and test V4 containers once AIM SDK is approved" -ForegroundColor Yellow
