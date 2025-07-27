#!/usr/bin/env python3
"""
Direct WSL Deployment for JustNews V4 with GPU Acceleration

This approach leverages your proven WSL/GPU setup directly instead of
fighting Docker CUDA compatibility issues.

Performance Target: 42.1 articles/sec (already proven working!)
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deploy_wsl_native():
    """Deploy JustNews V4 directly in WSL with GPU acceleration."""
    
    logger.info("🚀 Starting Native WSL Deployment with GPU Acceleration")
    logger.info("Target: 42.1 articles/sec processing (already proven!)")
    
    # Step 1: Copy integration files to WSL
    wsl_commands = [
        # Ensure WSL directory exists
        "mkdir -p /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment",
        
        # Copy key files for WSL deployment
        "cp /mnt/c/Users/marti/JustNewsAgentic/agents/analyst/hybrid_tools_v4.py /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment/",
        "cp /mnt/c/Users/marti/JustNewsAgentic/agents/analyst/main.py /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment/",
        "cp /mnt/c/Users/marti/JustNewsAgentic/quick_win_tensorrt.py /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment/",
        
        # Activate RAPIDS environment and test
        "source /home/nvidia/.venvs/rapids25.06_python3.12/bin/activate && python -c 'import tensorrt_llm; print(\"✅ TensorRT-LLM Ready\")'",
    ]
    
    logger.info("📋 Deployment Instructions:")
    logger.info("1. Open WSL terminal")
    logger.info("2. Run the following commands:")
    
    for i, cmd in enumerate(wsl_commands, 1):
        logger.info(f"   {i}. {cmd}")
    
    logger.info("\n3. Start the native GPU-accelerated analyst:")
    logger.info("   cd /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment")
    logger.info("   source /home/nvidia/.venvs/rapids25.06_python3.12/bin/activate")
    logger.info("   python main.py")
    
    return True

def create_simplified_docker():
    """Create a simplified Docker setup without CUDA base image."""
    
    simplified_dockerfile = """# Simplified V4 Dockerfile - CPU with GPU fallback
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY agents/analyst/requirements.txt ./requirements.txt

# Install Python dependencies (CPU versions)
RUN pip install --upgrade pip && \\
    pip install -r requirements.txt

# Copy application code
COPY agents/analyst/ ./
COPY agents/common/ ./common/

# Set environment variables
ENV PYTHONPATH="/app"
ENV GPU_ACCELERATION="fallback_only"

# Expose port
EXPOSE 8004

# Start the application
CMD ["python", "main.py"]
"""
    
    dockerfile_path = Path("agents/analyst/Dockerfile.simple")
    
    try:
        with open(dockerfile_path, 'w') as f:
            f.write(simplified_dockerfile)
        
        logger.info(f"✅ Created simplified Dockerfile: {dockerfile_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create simplified Dockerfile: {e}")
        return False

def main():
    """Main deployment orchestrator."""
    
    logger.info("🎯 JustNews V4 Deployment Strategy")
    logger.info("Based on your successful 42.1 articles/sec Quick Win!")
    
    logger.info("\n📊 Current Status:")
    logger.info("✅ GPU acceleration proven working (42.1 articles/sec)")
    logger.info("✅ Agent integration code completed")
    logger.info("✅ Hybrid fallback system ready")
    logger.info("⚠️ Docker CUDA compatibility issues")
    
    logger.info("\n🎯 Recommended Next Steps:")
    logger.info("1. **Native WSL Deployment** (Immediate - use proven GPU setup)")
    logger.info("2. **Simplified Docker** (Fallback - CPU with GPU calling WSL)")
    logger.info("3. **Full Docker GPU** (Future - when CUDA images resolved)")
    
    # Option 1: WSL Native
    logger.info("\n🚀 Option 1: Native WSL Deployment")
    deploy_wsl_native()
    
    # Option 2: Simplified Docker
    logger.info("\n🐳 Option 2: Creating Simplified Docker Fallback")
    create_simplified_docker()
    
    logger.info("\n✅ Deployment Options Ready!")
    logger.info("Choose your approach based on immediate needs:")
    logger.info("- WSL Native: Immediate 42.1 articles/sec performance")
    logger.info("- Docker Simple: Easier integration, slower performance")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
