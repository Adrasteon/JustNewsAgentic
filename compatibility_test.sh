#!/bin/bash
# JustNews V4 Python Compatibility Test Script

echo "=== Testing Python Version Compatibility ==="
PYTHON_VERSION=$(python3 --version)
echo "Python Version: $PYTHON_VERSION"
echo

# Test PyTorch installation
echo "Testing PyTorch..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" 2>/dev/null && echo "✅ PyTorch OK" || echo "❌ PyTorch FAILED"

# Test RAPIDS components
echo "Testing RAPIDS..."
python3 -c "import cudf; print(f'cuDF: {cudf.__version__}')" 2>/dev/null && echo "✅ RAPIDS cuDF OK" || echo "❌ RAPIDS cuDF FAILED"
python3 -c "import cuml; print(f'cuML: {cuml.__version__}')" 2>/dev/null && echo "✅ RAPIDS cuML OK" || echo "❌ RAPIDS cuML FAILED"

# Test TensorRT components
echo "Testing TensorRT..."
python3 -c "import tensorrt as trt; print(f'TensorRT: {trt.__version__}')" 2>/dev/null && echo "✅ TensorRT OK" || echo "❌ TensorRT FAILED"

# Test HuggingFace
echo "Testing HuggingFace..."
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null && echo "✅ HuggingFace OK" || echo "❌ HuggingFace FAILED"

# Test Sentence Transformers
echo "Testing Sentence-Transformers..."
python3 -c "import sentence_transformers; print(f'Sentence-Transformers: {sentence_transformers.__version__}')" 2>/dev/null && echo "✅ Sentence-Transformers OK" || echo "❌ Sentence-Transformers FAILED"

echo
echo "=== GPU Compatibility Test ==="
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'CUDA Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}')
else:
    print('❌ No CUDA GPU available')
"

echo
echo "=== Performance Baseline Test ==="
python3 -c "
import torch
import time
if torch.cuda.is_available():
    # Simple tensor operation benchmark
    device = torch.device('cuda')
    x = torch.randn(1000, 1000).to(device)
    start = time.time()
    for _ in range(100):
        y = torch.mm(x, x)
    torch.cuda.synchronize()
    end = time.time()
    print(f'GPU Matrix Multiplication: {(end-start)*1000:.2f}ms for 100 ops')
else:
    print('Skipping GPU performance test - no CUDA available')
"
