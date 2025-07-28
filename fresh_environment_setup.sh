#!/bin/bash

echo "🔥 Creating Fresh JustNews V4 Environment"
echo "========================================"

# Initialize conda
eval "$(conda shell.bash hook)"

echo "🗑️ Removing old environment completely..."
conda deactivate 2>/dev/null || true
conda env remove -n justnews-v4-optimized -y 2>/dev/null || true

echo "🆕 Creating brand new Python 3.12 environment..."
conda create -n justnews-v4-optimized python=3.12 -y

echo "🔧 Activating new environment..."
conda activate justnews-v4-optimized

echo "📦 Installing essential packages first..."
conda install -y numpy scipy scikit-learn matplotlib pandas

echo "🔥 Installing PyTorch ecosystem from conda-forge (more stable)..."
conda install -y pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

echo "✅ Testing PyTorch installation..."
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"

if [ $? -ne 0 ]; then
    echo "❌ Conda PyTorch failed. Trying pip as backup..."
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
fi

echo "🤗 Installing HuggingFace stack..."
pip install transformers==4.44.2 tokenizers==0.19.1

echo "📊 Installing sentence-transformers..."
pip install sentence-transformers

echo "🚀 Installing additional packages..."
pip install fastapi uvicorn requests

echo "🎯 Final comprehensive test..."
python -c "
try:
    import torch
    import torchvision
    import transformers
    import sentence_transformers
    
    print('🎉 SUCCESS - All packages imported!')
    print(f'   PyTorch: {torch.__version__}')
    print(f'   TorchVision: {torchvision.__version__}')
    print(f'   Transformers: {transformers.__version__}')
    print(f'   Sentence-Transformers: {sentence_transformers.__version__}')
    print(f'   CUDA Available: {torch.cuda.is_available()}')
    
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
        
    # Quick functionality test
    from transformers import pipeline
    print('✅ HuggingFace pipeline import successful')
    
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
"

echo ""
echo "✅ Fresh environment setup complete!"
echo "🔧 Environment name: justnews-v4-optimized"
echo "🚀 Ready for realistic performance testing!"
