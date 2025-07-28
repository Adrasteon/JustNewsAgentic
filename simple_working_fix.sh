#!/bin/bash

echo "🔧 Simple Working Fix - Using Available Versions"
echo "==============================================="

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate justnews-v4-optimized

echo "🧹 Cleaning up conflicts..."
pip uninstall -y sentence-transformers transformers tokenizers

echo "🔥 Installing PyTorch 2.2.0 + matching torchvision (available versions)..."
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 --index-url https://download.pytorch.org/whl/cu121

echo "✅ Testing PyTorch + TorchVision..."
python -c "
try:
    import torch
    import torchvision
    import torchvision.ops as ops
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ TorchVision: {torchvision.__version__}')
    print(f'✅ CUDA Available: {torch.cuda.is_available()}')
    print('✅ torchvision.ops imported successfully (nms operator working)')
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
"

echo "🤗 Installing compatible transformers..."
pip install transformers==4.39.0  # Compatible with PyTorch 2.2.0

echo "📊 Installing compatible sentence-transformers..."
pip install sentence-transformers==2.6.1  # Works with transformers 4.39.0

echo "🎯 Final test - All packages together..."
python -c "
try:
    import torch
    import torchvision
    import transformers
    import sentence_transformers
    from transformers import pipeline
    
    print('🎉 SUCCESS - All packages working!')
    print(f'   PyTorch: {torch.__version__}')
    print(f'   TorchVision: {torchvision.__version__}')
    print(f'   Transformers: {transformers.__version__}')
    print(f'   Sentence-Transformers: {sentence_transformers.__version__}')
    print(f'   CUDA Available: {torch.cuda.is_available()}')
    
    # Quick functional test
    classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    result = classifier('This is working great!')
    print(f'✅ Pipeline test: {result[0][\"label\"]}')
    
except Exception as e:
    print(f'❌ Final test failed: {e}')
    exit(1)
"

echo ""
echo "✅ Environment fixed with working versions!"
echo "🚀 Ready for realistic performance testing!"
