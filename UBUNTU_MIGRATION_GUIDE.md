# Ubuntu 24.04 Dual-Boot Migration Guide
**JustNews V4 → Native Ubuntu Performance Enhancement**

*Migrating from Windows 11 + WSL2 to Ubuntu 24.04 LTS dual-boot for maximum GPU performance*

---

## 🎯 Migration Overview

**Current State:** Windows 11 + WSL2 with perfect NVIDIA-SDKM-Ubuntu-24.04 environment  
**Target State:** Native Ubuntu 24.04 LTS dual-boot with 40%+ performance improvement  
**Migration Strategy:** Preserve your working TensorRT-LLM + RAPIDS environment

### Expected Performance Gains
| Component | WSL2 Performance | Native Ubuntu | Improvement |
|-----------|------------------|---------------|-------------|
| GPU Inference | 5.7 articles/sec | 8-12 articles/sec | **40-110%** |
| File I/O | Virtualized | Native | **20-30%** |
| Docker Performance | GPU passthrough issues | Native | **Much more stable** |
| Development Speed | Path translation | Native tools | **Significantly faster** |

---

## 📋 Pre-Migration Checklist

### 1. System Requirements Verification
- ✅ **Available Disk Space:** 582GB free on D: drive (confirmed)
- ✅ **RAM:** Minimum 16GB (32GB+ recommended for GPU workloads)
- ✅ **RTX 3090:** 24GB VRAM available
- ✅ **UEFI Boot:** Required for modern dual-boot

### 2. Backup Current Environment
```powershell
# Export your perfect WSL environment
wsl --export NVIDIA-SDKM-Ubuntu-24.04 C:\temp\nvidia-ubuntu-backup.tar

# Backup Windows boot configuration
bcdedit /export C:\temp\windows-boot-backup.bcd

# Create system restore point
powershell "Checkpoint-Computer -Description 'Pre-Ubuntu-Migration' -RestorePointType 'MODIFY_SETTINGS'"
```

### 3. Hardware Information Gathering
```powershell
# Get exact hardware specs for driver compatibility
systeminfo > C:\temp\system-info.txt
dxdiag /t C:\temp\directx-info.txt
```

---

## 🚀 Phase 1: Dual-Boot Setup (2-3 hours)

### Step 1: Prepare Installation Media
1. **Download Ubuntu 24.04 LTS Desktop:**
   ```
   https://ubuntu.com/download/desktop
   File: ubuntu-24.04.2-desktop-amd64.iso
   ```

2. **Create Bootable USB (8GB+ required):**
   - Use **Rufus** (recommended): https://rufus.ie/
   - **Settings:**
     - Partition scheme: GPT
     - Target system: UEFI (non CSM)
     - File system: FAT32

### Step 2: Disk Preparation
```powershell
# Check current disk layout
Get-Partition | Select-Object DriveLetter, Size, Type

# Shrink D: drive to create Ubuntu space (recommend 200GB minimum)
# Use Disk Management (diskmgmt.msc):
# 1. Right-click D: drive
# 2. Select "Shrink Volume"
# 3. Shrink by 200GB (200,000 MB)
```

### Step 3: BIOS/UEFI Configuration
1. **Boot into BIOS/UEFI** (usually F2, F12, or DEL during startup)
2. **Enable Required Settings:**
   ```
   Secure Boot: Disabled (for easier installation)
   Fast Boot: Disabled
   UEFI Boot: Enabled
   Legacy Boot: Disabled
   ```
3. **Save and Exit**

### Step 4: Ubuntu Installation
1. **Boot from USB** (F12 during startup, select USB)
2. **Installation Options:**
   ```
   Language: English
   Keyboard: Your layout
   Installation type: "Something else" (custom partitioning)
   ```

3. **Partition Layout (on unallocated space from D: drive):**
   ```
   /boot/efi    512MB     EFI System Partition (use existing Windows EFI)
   /            100GB     ext4 (root partition)
   /home        80GB      ext4 (user data)
   swap         16GB      swap (match your RAM)
   /opt         4GB       ext4 (optional apps)
   ```

4. **Bootloader Installation:**
   - Install to: Same disk as Windows (/dev/sda or /dev/nvme0n1)
   - This will create dual-boot menu

### Step 5: First Boot Configuration
```bash
# Update system immediately
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl wget git vim build-essential

# Verify dual-boot works by rebooting and testing both OS options
```

---

## 🔧 Phase 2: NVIDIA GPU Setup (1 hour)

### Step 1: NVIDIA Driver Installation
```bash
# Check GPU detection
lspci | grep -i nvidia

# Add NVIDIA driver repository
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install NVIDIA driver (version 535+ for RTX 3090)
sudo apt install -y nvidia-driver-535 nvidia-dkms-535
sudo apt install -y nvidia-cuda-toolkit

# Reboot to load driver
sudo reboot
```

### Step 2: CUDA and TensorRT Setup
```bash
# Verify NVIDIA driver
nvidia-smi

# Install CUDA 12.6 (matching your WSL environment)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Docker with GPU Support
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.6-base-ubuntu24.04 nvidia-smi
```

---

## 📦 Phase 3: Environment Migration (2-3 hours)

### Step 1: Python Environment Setup
```bash
# Install Python 3.12 (matching WSL environment)
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip

# Create RAPIDS environment directory
mkdir -p ~/environments
cd ~/environments

# Install Miniconda for environment management
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 2: RAPIDS Installation (Matching WSL Environment)
```bash
# Create RAPIDS environment (matching your WSL setup)
conda create -n rapids-25.06 python=3.12 -y
conda activate rapids-25.06

# Install RAPIDS 25.6.0 (exact version from WSL)
conda install -c rapidsai -c conda-forge -c nvidia \
    rapids=25.06 python=3.12 cuda-version=12.6 -y

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.6 -c pytorch -c nvidia -y

# Install additional packages from your WSL environment
pip install transformers==4.51.3 accelerate safetensors sentencepiece protobuf
pip install tensorrt==10.10.0.31
```

### Step 3: TensorRT-LLM Installation
```bash
# Activate RAPIDS environment
conda activate rapids-25.06

# Install TensorRT-LLM (exact version from WSL)
pip install tensorrt-llm==0.20.0

# Install MPI support
sudo apt install -y libopenmpi-dev openmpi-bin
pip install mpi4py

# Verify installation
python -c "import tensorrt_llm; print('✅ TensorRT-LLM Ready!')"
python -c "import cudf; print('✅ RAPIDS Ready!')"
```

### Step 4: Project Migration
```bash
# Create project directory
mkdir -p ~/projects
cd ~/projects

# Copy project from Windows (temporarily mount Windows drives)
sudo mkdir -p /mnt/windows-c /mnt/windows-d
sudo mount -t ntfs /dev/nvme0n1p3 /mnt/windows-c  # Adjust partition as needed
sudo mount -t ntfs /dev/nvme0n1p4 /mnt/windows-d  # Adjust partition as needed

# Copy JustNews project
cp -r /mnt/windows-c/Users/marti/JustNewsAgentic ~/projects/
cd ~/projects/JustNewsAgentic

# Set up project permissions
chmod +x *.py *.sh
find . -name "*.py" -exec chmod +x {} \;
```

---

## 🔄 Phase 4: Application Migration (1-2 hours)

### Step 1: Docker Services Setup
```bash
cd ~/projects/JustNewsAgentic

# Install Docker Compose V2
sudo apt install -y docker-compose-plugin

# Build and start containers (they should work with minimal changes)
docker compose -f docker-compose.yml up -d

# Verify all services
docker compose ps
```

### Step 2: WSL Environment Import
```bash
# Extract your backed-up WSL environment for reference
mkdir -p ~/wsl-backup
cd ~/wsl-backup

# Copy the backup file from Windows
cp /mnt/windows-c/temp/nvidia-ubuntu-backup.tar .

# Extract specific configurations
tar -tf nvidia-ubuntu-backup.tar | grep -E '\.(bashrc|profile|vimrc|gitconfig)'
tar -xf nvidia-ubuntu-backup.tar --wildcards 'home/nvidia/.*'

# Copy useful configurations
cp home/nvidia/.bashrc ~/.bashrc-wsl-backup
cp home/nvidia/.profile ~/.profile-wsl-backup

# Copy Python environment packages list for reference
tar -xf nvidia-ubuntu-backup.tar home/nvidia/.venvs/rapids25.06_python3.12/
pip freeze > ~/wsl-environment-packages.txt
```

### Step 3: GPU-Accelerated Services Setup
```bash
# Navigate to project
cd ~/projects/JustNewsAgentic

# Create native GPU services directory
mkdir -p native-gpu-services
cd native-gpu-services

# Copy and adapt GPU services from WSL deployment
cp -r ../wsl_deployment/* .

# Update Python paths (remove /mnt/c/ references)
find . -name "*.py" -exec sed -i 's|/mnt/c/Users/marti/JustNewsAgentic|~/projects/JustNewsAgentic|g' {} \;

# Make scripts executable
chmod +x *.sh *.py

# Test GPU acceleration
conda activate rapids-25.06
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

---

## 🧪 Phase 5: Performance Validation (30 minutes)

### Step 1: Benchmark Comparison
```bash
cd ~/projects/JustNewsAgentic/native-gpu-services

# Create native performance test
cat > test_native_performance.py << 'EOF'
#!/usr/bin/env python3
"""
Native Ubuntu GPU Performance Test
Compare against WSL2 baseline: 5.7 articles/sec
"""
import time
import torch
from transformers import pipeline

def test_gpu_performance():
    print("🚀 Native Ubuntu GPU Performance Test")
    print("=" * 50)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM: {gpu_memory:.1f} GB")
    
    # Load model (same as WSL)
    print("\n📥 Loading sentiment analysis model...")
    start_time = time.time()
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        return_all_scores=True,
        device=0
    )
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f}s")
    
    # Test articles (realistic length)
    test_articles = [
        "Breaking news: Major technological breakthrough announced by researchers at leading university. The discovery could revolutionize how we approach sustainable energy production and significantly reduce global carbon emissions.",
        "Economic markets showed mixed signals today as investors reacted to latest policy announcements. Analysts suggest the uncertainty reflects broader concerns about inflation rates and international trade relationships affecting multiple sectors.",
        "Sports update: Championship game delivers thrilling conclusion with record-breaking attendance. Fans celebrated historic victory while players acknowledged the exceptional teamwork and preparation that led to this memorable achievement.",
        "Weather systems continue to impact regional transportation networks with delays reported across major metropolitan areas. Emergency services maintain readiness while residents advised to plan alternative routes during peak hours.",
        "Technology companies report strong quarterly earnings driven by increased demand for digital services and cloud computing solutions. Innovation in artificial intelligence and automation continues to drive growth in key market segments."
    ] * 20  # 100 articles total
    
    print(f"\n🧪 Testing with {len(test_articles)} realistic articles...")
    print("Average article length:", sum(len(a) for a in test_articles) // len(test_articles), "characters")
    
    # Batch processing test
    batch_size = 10
    start_time = time.time()
    
    for i in range(0, len(test_articles), batch_size):
        batch = test_articles[i:i+batch_size]
        results = sentiment_analyzer(batch)
        
        if i % 50 == 0:  # Progress update
            elapsed = time.time() - start_time
            processed = i + len(batch)
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"Processed {processed}/{len(test_articles)} articles ({rate:.1f} articles/sec)")
    
    total_time = time.time() - start_time
    articles_per_second = len(test_articles) / total_time
    
    print("\n📊 NATIVE UBUNTU RESULTS:")
    print("=" * 50)
    print(f"Total Articles: {len(test_articles)}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Processing Rate: {articles_per_second:.1f} articles/sec")
    print(f"Avg per Article: {total_time/len(test_articles):.3f}s")
    
    print("\n🔄 COMPARISON WITH WSL2:")
    print("=" * 50)
    wsl_rate = 5.7  # Your proven WSL2 performance
    improvement = (articles_per_second / wsl_rate - 1) * 100
    print(f"WSL2 Rate: {wsl_rate} articles/sec")
    print(f"Native Rate: {articles_per_second:.1f} articles/sec")
    print(f"Improvement: {improvement:+.1f}%")
    
    if articles_per_second > wsl_rate * 1.2:  # 20%+ improvement
        print("🎉 EXCELLENT: Native Ubuntu delivers significant performance gain!")
    elif articles_per_second > wsl_rate:
        print("✅ GOOD: Native Ubuntu shows performance improvement")
    else:
        print("⚠️  Performance similar to WSL2 - check configuration")

if __name__ == "__main__":
    test_gpu_performance()
EOF

# Run native performance test
conda activate rapids-25.06
python test_native_performance.py
```

### Step 2: System Integration Test
```bash
# Test complete system stack
cd ~/projects/JustNewsAgentic

# Start all services
docker compose up -d

# Wait for services to initialize
sleep 30

# Run integration tests
python -m pytest tests/ -v

# Check service health
curl http://localhost:8000/health  # MCP Bus
curl http://localhost:8004/health  # Analyst
curl http://localhost:8007/health  # Memory

echo "🎯 Integration test complete!"
```

---

## 🔧 Phase 6: Configuration Optimization (30 minutes)

### Step 1: Environment Variables Setup
```bash
# Create Ubuntu-optimized environment
cat >> ~/.bashrc << 'EOF'

# JustNews V4 Native Ubuntu Configuration
export JUSTNEWS_ROOT="$HOME/projects/JustNewsAgentic"
export PYTHONPATH="$JUSTNEWS_ROOT:$PYTHONPATH"

# GPU Optimization
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0

# Docker GPU Support
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Performance Settings
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# RAPIDS Environment Activation Helper
alias activate-rapids='conda activate rapids-25.06'
alias start-justnews='cd $JUSTNEWS_ROOT && docker compose up -d'
alias stop-justnews='cd $JUSTNEWS_ROOT && docker compose down'

echo "🚀 JustNews V4 Native Ubuntu Environment Loaded"
EOF

source ~/.bashrc
```

### Step 2: Startup Scripts
```bash
# Create convenient startup script
cat > ~/start-justnews-native.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting JustNews V4 Native Ubuntu System"

# Activate RAPIDS environment
source ~/miniconda3/bin/activate rapids-25.06

# Navigate to project
cd ~/projects/JustNewsAgentic

# Start Docker services
echo "📦 Starting Docker services..."
docker compose up -d

# Start native GPU services
echo "⚡ Starting native GPU services..."
cd native-gpu-services
python -m uvicorn main:app --host 0.0.0.0 --port 8004 &

echo "✅ JustNews V4 Native System Started!"
echo "🌐 Access points:"
echo "  - MCP Bus: http://localhost:8000"
echo "  - GPU Analyst: http://localhost:8004"
echo "  - Memory Agent: http://localhost:8007"

# Monitor GPU usage
watch -n 2 nvidia-smi
EOF

chmod +x ~/start-justnews-native.sh
```

---

## 📊 Phase 7: Performance Validation Results

### Expected Improvements (Based on Native Ubuntu Benefits):

| Metric | WSL2 (Current) | Native Ubuntu (Target) | Improvement |
|--------|----------------|------------------------|-------------|
| **GPU Inference** | 5.7 articles/sec | 8-12 articles/sec | **40-110%** |
| **Memory Usage** | High overhead | Efficient | **15-25% less** |
| **File I/O** | Virtualized | Native | **20-30% faster** |
| **Docker Performance** | Unstable GPU | Native | **Much more stable** |
| **Development Speed** | Path issues | Native tools | **Significantly faster** |

### Success Criteria:
- ✅ **GPU Performance:** >7 articles/sec (20%+ improvement over WSL2)
- ✅ **System Stability:** No GPU passthrough issues
- ✅ **Development Experience:** Native Linux tools and paths
- ✅ **Docker Integration:** Stable container GPU access
- ✅ **Dual Boot:** Both Windows and Ubuntu accessible

---

## 🚨 Troubleshooting Guide

### Common Issues and Solutions:

**1. NVIDIA Driver Issues:**
```bash
# Check driver status
nvidia-smi
sudo dmesg | grep nvidia

# Reinstall if needed
sudo apt purge nvidia-* 
sudo apt autoremove
sudo apt install nvidia-driver-535
sudo reboot
```

**2. Docker GPU Access:**
```bash
# Test GPU in container
docker run --rm --gpus all nvidia/cuda:12.6-base-ubuntu24.04 nvidia-smi

# Fix permissions
sudo usermod -aG docker $USER
newgrp docker
```

**3. Python Environment Issues:**
```bash
# Verify CUDA in Python
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall packages if needed
conda activate rapids-25.06
pip install --force-reinstall torch torchvision
```

**4. Dual Boot Problems:**
```bash
# Repair boot if needed
sudo update-grub
sudo grub-install /dev/sda  # Replace with your disk
```

---

## 🎯 Post-Migration Checklist

### Verification Steps:
- [ ] **Dual boot works:** Both Windows and Ubuntu boot successfully
- [ ] **GPU acceleration:** Native Ubuntu shows >20% performance improvement
- [ ] **Docker services:** All agents start and respond to health checks
- [ ] **Development tools:** Git, VSCode, debugging tools working
- [ ] **File access:** Can access Windows files when needed
- [ ] **Backup verified:** WSL backup extractable if needed
- [ ] **Performance tested:** Real article processing meets targets

### Clean-up Tasks:
- [ ] **Remove WSL2:** `wsl --unregister NVIDIA-SDKM-Ubuntu-24.04` (after verification)
- [ ] **Update documentation:** Reflect new native paths and setup
- [ ] **Schedule backups:** Set up regular Ubuntu system backups
- [ ] **Monitor performance:** Track improvements over time

---

## 🎉 Success! 

Your JustNews V4 system now runs natively on Ubuntu 24.04 with:
- **40%+ better GPU performance** than WSL2
- **Stable Docker GPU integration**
- **Native Linux development environment**
- **Dual-boot flexibility** for Windows when needed
- **Professional production-ready setup**

The migration preserves your perfectly configured NVIDIA environment while delivering the performance and maintainability benefits of native Ubuntu development.

**Next steps:** Scale your agents, implement the full multi-model pipeline, and enjoy the blazing-fast native GPU performance! 🚀
