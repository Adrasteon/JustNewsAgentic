#!/bin/bash
# Fix PyTorch/TorchVision/Transformers Compatibility Issues

echo "🔧 Fixing PyTorch ecosystem compatibility issues..."
echo

# Make sure we're in the right environment
conda activate justnews-v4-optimized

echo "📦 Uninstalling problematic packages..."
pip uninstall -y torch torchvision torchaudio transformers sentence-transformers

echo "🔥 Installing compatible PyTorch ecosystem (CUDA 12.1)..."
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

echo "🤗 Installing compatible HuggingFace transformers..."
pip install transformers==4.44.2

echo "📊 Installing sentence-transformers..."
pip install sentence-transformers

echo "✅ Testing installations..."
echo "Testing PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

echo "Testing Transformers..."
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo "Testing Sentence-Transformers..."
python -c "import sentence_transformers; print(f'Sentence-Transformers: {sentence_transformers.__version__}')"

echo
echo "🎉 Compatibility issues resolved!"
