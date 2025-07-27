#!/bin/bash
# WSL Native JustNews V4 GPU Deployment Script

echo "Starting JustNews V4 with GPU Acceleration"
echo "Target: 42.1 articles/sec processing!"

# Navigate to deployment directory
cd /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment

# Activate RAPIDS environment with TensorRT-LLM
echo "Activating GPU environment..."
source /home/nvidia/.venvs/rapids25.06_python3.12/bin/activate

# Verify GPU setup
echo "Verifying GPU setup..."
python -c "import tensorrt_llm; print('TensorRT-LLM Ready')" || {
    echo "TensorRT-LLM not available, check environment"
    exit 1
}

python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')" || {
    echo "PyTorch CUDA not available"
    exit 1
}

# Check NVIDIA GPU
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

echo "Starting GPU-accelerated analyst agent..."
echo "Expected performance: 42.1 articles/sec"
echo "API endpoint: http://localhost:8004"

# Start the analyst with GPU acceleration
python main.py
