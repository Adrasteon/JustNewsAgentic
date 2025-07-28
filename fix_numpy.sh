#!/bin/bash

echo "🔧 Fixing NumPy Compatibility Issue"
echo "=================================="

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate justnews-v4-optimized

echo "📦 Current NumPy version:"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

echo "⬇️ Downgrading to NumPy 1.x (compatible with PyTorch 2.2.0)..."
pip install "numpy<2.0" --force-reinstall

echo "✅ Testing NumPy downgrade..."
python -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')"

echo "🎯 Testing complete PyTorch ecosystem..."
python -c "
try:
    import numpy
    import torch
    import torchvision  
    import transformers
    import sentence_transformers
    from transformers import pipeline
    
    print('🎉 ALL PACKAGES WORKING PERFECTLY!')
    print(f'   NumPy: {numpy.__version__}')
    print(f'   PyTorch: {torch.__version__}')
    print(f'   TorchVision: {torchvision.__version__}')
    print(f'   Transformers: {transformers.__version__}')
    print(f'   Sentence-Transformers: {sentence_transformers.__version__}')
    print(f'   CUDA Available: {torch.cuda.is_available()}')
    
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
    
    # Test actual functionality
    print('\\n🧪 Quick functionality test...')
    classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    result = classifier('This environment is working perfectly!')
    print(f'✅ Sentiment pipeline: {result[0][\"label\"]} ({result[0][\"score\"]:.3f})')
    
    print('\\n🚀 Environment ready for realistic performance testing!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo ""
echo "✅ NumPy compatibility fixed!"
echo "🎯 Ready to run realistic_performance_test.py!"
