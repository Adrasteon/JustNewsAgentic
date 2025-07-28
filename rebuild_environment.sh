#!/bin/bash

echo "🔥 Complete Environment Rebuild for JustNews V4"
echo "=============================================="

# Activate the environment (ensure conda is initialized)
eval "$(conda shell.bash hook)"
conda activate justnews-v4-optimized

echo "🧹 Completely cleaning PyTorch ecosystem..."
pip uninstall -y torch torchvision torchaudio transformers sentence-transformers tokenizers triton
pip uninstall -y nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12
pip uninstall -y nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12
pip uninstall -y nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12

echo "🧼 Clearing pip cache..."
pip cache purge

echo "📦 Installing PyTorch 2.4.1 with CUDA 12.1 (fresh install)..."
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

echo "✅ Testing PyTorch installation..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} - CUDA Available: {torch.cuda.is_available()}')"

if [ $? -ne 0 ]; then
    echo "❌ PyTorch installation failed. Trying alternative approach..."
    
    # Alternative: Use conda-forge PyTorch
    echo "🔄 Trying conda-forge PyTorch..."
    conda install -y pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    
    echo "✅ Testing conda PyTorch..."
    python -c "import torch; print(f'✅ PyTorch {torch.__version__} - CUDA Available: {torch.cuda.is_available()}')"
fi

echo "🤗 Installing HuggingFace transformers (compatible version)..."
pip install transformers==4.44.2

echo "✅ Testing transformers..."
python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')"

echo "📊 Installing sentence-transformers..."
pip install sentence-transformers

echo "✅ Testing sentence-transformers..."
python -c "import sentence_transformers; print(f'✅ Sentence-Transformers {sentence_transformers.__version__}')"

echo "🎯 Final compatibility test..."
python -c "
import torch
import transformers
import sentence_transformers

print('🎉 All packages imported successfully!')
print(f'   PyTorch: {torch.__version__}')
print(f'   Transformers: {transformers.__version__}')
print(f'   Sentence-Transformers: {sentence_transformers.__version__}')
print(f'   CUDA Available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'   CUDA Version: {torch.version.cuda}')
    print(f'   GPU Device: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "✅ Environment rebuild complete!"
echo "📍 Active environment: justnews-v4-optimized"
echo "🚀 Ready for JustNews V4 deployment!"
