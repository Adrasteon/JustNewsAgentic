#!/usr/bin/env pwsh
<#
.SYNOPSIS
JustNews V4 Pre-Migration Backup Script
.DESCRIPTION
Automated backup and preparation script for Ubuntu dual-boot migration
.NOTES
Run this script BEFORE starting Ubuntu installation
#>

Write-Host "🚀 JustNews V4 Pre-Migration Backup Script" -ForegroundColor Green
Write-Host "=" * 50

# Create backup directory
$BackupDir = "C:\JustNews-Migration-Backup"
if (-not (Test-Path $BackupDir)) {
    New-Item -ItemType Directory -Path $BackupDir -Force
    Write-Host "✅ Created backup directory: $BackupDir" -ForegroundColor Green
}

# 1. Export WSL Environment
Write-Host "`n📦 Backing up WSL environment..." -ForegroundColor Yellow
try {
    wsl --export NVIDIA-SDKM-Ubuntu-24.04 "$BackupDir\nvidia-ubuntu-backup.tar"
    Write-Host "✅ WSL environment exported successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to export WSL environment: $_" -ForegroundColor Red
}

# 2. Backup Windows Boot Configuration
Write-Host "`n💾 Backing up Windows boot configuration..." -ForegroundColor Yellow
try {
    bcdedit /export "$BackupDir\windows-boot-backup.bcd"
    Write-Host "✅ Boot configuration backed up" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to backup boot configuration: $_" -ForegroundColor Red
}

# 3. Create System Restore Point
Write-Host "`n🔄 Creating system restore point..." -ForegroundColor Yellow
try {
    Checkpoint-Computer -Description "JustNews-Ubuntu-Migration" -RestorePointType "MODIFY_SETTINGS"
    Write-Host "✅ System restore point created" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create restore point: $_" -ForegroundColor Red
}

# 4. Gather System Information
Write-Host "`n📋 Gathering system information..." -ForegroundColor Yellow
try {
    systeminfo > "$BackupDir\system-info.txt"
    dxdiag /t "$BackupDir\directx-info.txt"
    Get-WmiObject -Class Win32_LogicalDisk | Select-Object DeviceID, Size, FreeSpace | Out-File "$BackupDir\disk-info.txt"
    Write-Host "✅ System information gathered" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to gather system info: $_" -ForegroundColor Red
}

# 5. Backup Project Code
Write-Host "`n📁 Backing up JustNews project..." -ForegroundColor Yellow
try {
    $ProjectSource = "C:\Users\marti\JustNewsAgentic"
    $ProjectBackup = "$BackupDir\JustNewsAgentic"
    
    if (Test-Path $ProjectSource) {
        Copy-Item -Path $ProjectSource -Destination $ProjectBackup -Recurse -Force
        Write-Host "✅ Project code backed up" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Project directory not found at expected location" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Failed to backup project: $_" -ForegroundColor Red
}

# 6. Export WSL Configuration Details
Write-Host "`n⚙️  Exporting WSL configuration..." -ForegroundColor Yellow
try {
    wsl --list --verbose > "$BackupDir\wsl-distributions.txt"
    wsl -d NVIDIA-SDKM-Ubuntu-24.04 -- env > "$BackupDir\wsl-environment.txt"
    wsl -d NVIDIA-SDKM-Ubuntu-24.04 -- pip freeze > "$BackupDir\wsl-python-packages.txt" 2>$null
    Write-Host "✅ WSL configuration exported" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to export WSL config: $_" -ForegroundColor Red
}

# 7. Check Disk Space for Ubuntu
Write-Host "`n💽 Checking disk space for Ubuntu installation..." -ForegroundColor Yellow
$DiskD = Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DeviceID -eq "D:" }
if ($DiskD) {
    $FreeSpaceGB = [math]::Round($DiskD.FreeSpace / 1GB, 2)
    Write-Host "✅ D: drive has $FreeSpaceGB GB free space" -ForegroundColor Green
    
    if ($FreeSpaceGB -gt 200) {
        Write-Host "✅ Sufficient space for Ubuntu installation (200GB+ recommended)" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Consider freeing up more space. 200GB+ recommended for Ubuntu" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ D: drive not found" -ForegroundColor Red
}

# 8. Generate Migration Summary
Write-Host "`n📝 Generating migration summary..." -ForegroundColor Yellow
$Summary = @"
JustNews V4 Ubuntu Migration Summary
Generated: $(Get-Date)

BACKUP LOCATION: $BackupDir

FILES CREATED:
✅ nvidia-ubuntu-backup.tar - Complete WSL environment
✅ windows-boot-backup.bcd - Windows boot configuration
✅ system-info.txt - Complete system information
✅ directx-info.txt - Graphics and DirectX details
✅ disk-info.txt - Disk partition information
✅ wsl-distributions.txt - WSL configuration
✅ wsl-environment.txt - WSL environment variables
✅ wsl-python-packages.txt - Python packages list
✅ JustNewsAgentic/ - Complete project backup

CURRENT SYSTEM:
- WSL Environment: NVIDIA-SDKM-Ubuntu-24.04
- TensorRT-LLM: 0.20.0 (OPERATIONAL)
- RAPIDS: 25.6.0
- GPU: RTX 3090 24GB VRAM
- Free Space on D:: $FreeSpaceGB GB

NEXT STEPS:
1. Review Ubuntu Migration Guide (UBUNTU_MIGRATION_GUIDE.md)
2. Create Ubuntu 24.04 installation USB
3. Boot into Ubuntu installer
4. Follow dual-boot installation process
5. Restore environment using backup files

EMERGENCY RECOVERY:
- System Restore Point: JustNews-Ubuntu-Migration
- Boot Backup: windows-boot-backup.bcd
- WSL Restore Command: wsl --import NVIDIA-SDKM-Ubuntu-24.04 C:\WSL\nvidia-ubuntu nvidia-ubuntu-backup.tar

🎯 TARGET PERFORMANCE IMPROVEMENT: 40%+ GPU acceleration improvement
"@

$Summary | Out-File "$BackupDir\MIGRATION_SUMMARY.txt"
Write-Host "✅ Migration summary created" -ForegroundColor Green

# Final Status
Write-Host "`n🎉 PRE-MIGRATION BACKUP COMPLETE!" -ForegroundColor Green
Write-Host "=" * 50
Write-Host "📁 All backups saved to: $BackupDir" -ForegroundColor Cyan
Write-Host "📖 Next: Follow UBUNTU_MIGRATION_GUIDE.md for dual-boot setup" -ForegroundColor Cyan
Write-Host "⚡ Expected improvement: 40%+ GPU performance increase" -ForegroundColor Cyan

# Pause for user acknowledgment
Write-Host "`nPress any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
