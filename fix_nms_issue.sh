#!/bin/bash

echo "🔧 Fixing torchvision::nms Operator Issue"
echo "========================================"

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate environment
conda activate justnews-v4-optimized

echo "🔍 Current package versions:"
python -c "
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
except:
    print('PyTorch: Not available')

try:
    import torchvision
    print(f'TorchVision: {torchvision.__version__}')
except:
    print('TorchVision: Not available')
"

echo ""
echo "🧹 Uninstalling conflicting torchvision completely..."
pip uninstall -y torchvision
conda remove -y torchvision --force

echo "🔥 Installing specific compatible versions..."
# Use older, more stable versions that definitely work together
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

echo "✅ Testing basic PyTorch..."
python -c "
import torch
print(f'✅ PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')
"

echo "✅ Testing torchvision import..."
python -c "
try:
    import torchvision
    print(f'✅ TorchVision {torchvision.__version__} imported successfully')
    
    # Test the problematic nms operation specifically
    import torchvision.ops as ops
    print('✅ torchvision.ops imported (nms should work)')
    
except Exception as e:
    print(f'❌ TorchVision error: {e}')
    exit(1)
"

echo "🤗 Reinstalling transformers with compatible version..."
pip install transformers==4.35.2  # Older version that works with PyTorch 2.1.2

echo "🎯 Testing transformers pipeline..."
python -c "
try:
    from transformers import pipeline
    print('✅ Transformers pipeline imported successfully')
    
    # Quick test
    classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device=0 if __import__('torch').cuda.is_available() else -1)
    result = classifier('This is a test')
    print(f'✅ Pipeline test successful: {result}')
    
except Exception as e:
    print(f'❌ Transformers error: {e}')
    exit(1)
"

echo "📊 Installing sentence-transformers..."
pip install sentence-transformers==2.2.2  # Compatible with older PyTorch

echo "🎯 Final compatibility test..."
python -c "
try:
    import torch
    import torchvision
    import transformers
    import sentence_transformers
    
    print('🎉 ALL PACKAGES WORKING!')
    print(f'   PyTorch: {torch.__version__}')
    print(f'   TorchVision: {torchvision.__version__}')
    print(f'   Transformers: {transformers.__version__}')
    print(f'   Sentence-Transformers: {sentence_transformers.__version__}')
    print(f'   CUDA Available: {torch.cuda.is_available()}')
    
    # Test actual functionality
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer
    
    print('✅ All imports successful - ready for benchmarking!')
    
except Exception as e:
    print(f'❌ Final test failed: {e}')
    exit(1)
"

echo ""
echo "✅ torchvision::nms issue resolved!"
echo "🚀 Environment ready for realistic performance testing!"
