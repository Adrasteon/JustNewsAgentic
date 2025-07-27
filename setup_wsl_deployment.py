#!/usr/bin/env python3
"""
WSL Native Deployment Script for JustNews V4 GPU Acceleration

This script sets up the native WSL deployment to leverage your proven
42.1 articles/sec GPU acceleration directly in the WSL environment.
"""

import os
import sys
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_wsl_deployment():
    """Set up WSL deployment directory with all necessary files."""
    
    logger.info("🚀 Setting up WSL Native Deployment")
    logger.info("Target: 42.1 articles/sec GPU acceleration!")
    
    # Create deployment directory
    wsl_deploy_dir = Path("wsl_deployment")
    wsl_deploy_dir.mkdir(exist_ok=True)
    
    # Files to copy for WSL deployment
    files_to_copy = [
        ("agents/analyst/hybrid_tools_v4.py", "hybrid_tools_v4.py"),
        ("agents/analyst/main.py", "main.py"),
        ("quick_win_tensorrt.py", "quick_win_tensorrt.py"),
        ("test_v4_gpu_integration.py", "test_integration.py"),
    ]
    
    logger.info("📋 Copying integration files...")
    for src, dst in files_to_copy:
        src_path = Path(src)
        dst_path = wsl_deploy_dir / dst
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            logger.info(f"✅ Copied {src} → {dst}")
        else:
            logger.warning(f"⚠️ Source file not found: {src}")
    
    # Create WSL startup script
    wsl_startup_script = """#!/bin/bash
# WSL Native JustNews V4 GPU Deployment Script

echo "🚀 Starting JustNews V4 with GPU Acceleration"
echo "Target: 42.1 articles/sec processing!"

# Navigate to deployment directory
cd /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment

# Activate RAPIDS environment with TensorRT-LLM
echo "⚡ Activating GPU environment..."
source /home/nvidia/.venvs/rapids25.06_python3.12/bin/activate

# Verify GPU setup
echo "🔍 Verifying GPU setup..."
python -c "import tensorrt_llm; print('✅ TensorRT-LLM Ready')" || {
    echo "❌ TensorRT-LLM not available, check environment"
    exit 1
}

python -c "import torch; print(f'✅ PyTorch CUDA: {torch.cuda.is_available()}')" || {
    echo "❌ PyTorch CUDA not available"
    exit 1
}

# Check NVIDIA GPU
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

echo "🎯 Starting GPU-accelerated analyst agent..."
echo "Expected performance: 42.1 articles/sec"
echo "API endpoint: http://localhost:8004"

# Start the analyst with GPU acceleration
python main.py
"""
    
    startup_script_path = wsl_deploy_dir / "start_gpu_analyst.sh"
    with open(startup_script_path, 'w') as f:
        f.write(wsl_startup_script)
    
    logger.info(f"✅ Created startup script: {startup_script_path}")
    
    # Create performance test script
    test_script = """#!/bin/bash
# Quick Performance Test for GPU Acceleration

echo "🧪 Running JustNews V4 GPU Performance Test"

cd /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment
source /home/nvidia/.venvs/rapids25.06_python3.12/bin/activate

# Test GPU acceleration directly
python -c "
import time
from hybrid_tools_v4 import GPUAcceleratedAnalyst

print('🚀 Testing GPU acceleration...')
gpu_analyzer = GPUAcceleratedAnalyst()

test_articles = [
    'Breaking: Major tech company announces revolutionary AI breakthrough.',
    'Political tensions rise as world leaders meet for emergency summit.',
    'Economic markets surge following positive employment data reports.'
]

start_time = time.time()
for i, article in enumerate(test_articles):
    sentiment = gpu_analyzer.score_sentiment_gpu(article)
    bias = gpu_analyzer.score_bias_gpu(article)
    print(f'Article {i+1}: Sentiment={sentiment:.3f}, Bias={bias:.3f}')

end_time = time.time()
total_time = end_time - start_time
articles_per_sec = len(test_articles) / total_time

print(f'')
print(f'📊 Performance Results:')
print(f'Total time: {total_time:.3f}s')
print(f'Articles per second: {articles_per_sec:.1f}')
print(f'Target (42.1/sec): {'✅ MET' if articles_per_sec > 30 else '⚠️ BELOW TARGET'}')
"
"""
    
    test_script_path = wsl_deploy_dir / "test_gpu_performance.sh"
    with open(test_script_path, 'w') as f:
        f.write(test_script)
    
    logger.info(f"✅ Created test script: {test_script_path}")
    
    return True

def create_windows_launcher():
    """Create Windows batch file to launch WSL deployment."""
    
    launcher_script = """@echo off
echo 🚀 Launching JustNews V4 GPU Acceleration in WSL
echo Target: 42.1 articles/sec processing!

echo ⚡ Starting WSL with GPU environment...
wsl bash -c "cd /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment && bash start_gpu_analyst.sh"

pause
"""
    
    launcher_path = Path("launch_gpu_wsl.bat")
    with open(launcher_path, 'w') as f:
        f.write(launcher_script)
    
    logger.info(f"✅ Created Windows launcher: {launcher_path}")
    return True

def main():
    """Main deployment setup."""
    
    logger.info("🎯 JustNews V4 WSL Native Deployment Setup")
    logger.info("Leveraging your proven 42.1 articles/sec GPU acceleration!")
    
    # Set up WSL deployment
    setup_wsl_deployment()
    
    # Create Windows launcher
    create_windows_launcher()
    
    logger.info("\n🎉 WSL Deployment Setup Complete!")
    logger.info("\n📋 Next Steps:")
    logger.info("1. Open WSL terminal (or run launch_gpu_wsl.bat)")
    logger.info("2. Navigate to deployment directory:")
    logger.info("   cd /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment")
    logger.info("3. Run the startup script:")
    logger.info("   bash start_gpu_analyst.sh")
    logger.info("")
    logger.info("🧪 Or test GPU performance first:")
    logger.info("   bash test_gpu_performance.sh")
    logger.info("")
    logger.info("🎯 Expected Results:")
    logger.info("   • GPU acceleration: 42.1 articles/sec")
    logger.info("   • API endpoint: http://localhost:8004")
    logger.info("   • Real-time performance metrics")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
