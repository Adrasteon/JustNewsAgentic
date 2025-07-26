#!/bin/bash
# TensorRT-LLM Setup Script for JustNews V4
# This script installs and configures TensorRT-LLM in the RAPIDS environment
# Run this in WSL2 Ubuntu: wsl -d NVIDIA-SDKM-Ubuntu-24.04

set -e  # Exit on any error

echo "🚀 Setting up TensorRT-LLM for JustNews V4"
echo "Environment: NVIDIA-SDKM-Ubuntu-24.04"
echo "Target: RTX 3090 with RAPIDS 25.6.0"
echo "=========================================="

# Activate RAPIDS environment
echo "📦 Activating RAPIDS environment..."
source /home/nvidia/.venvs/rapids25.06_python3.12/bin/activate

# Verify CUDA availability
echo "🔍 Verifying CUDA installation..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Install TensorRT-LLM dependencies
echo "📦 Installing TensorRT-LLM dependencies..."
pip install --upgrade pip setuptools wheel

# Install TensorRT-LLM (official NVIDIA package)
echo "📦 Installing TensorRT-LLM..."
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com

# Install additional dependencies for RTX AI Toolkit integration
echo "📦 Installing RTX AI Toolkit dependencies..."
pip install transformers datasets accelerate peft
pip install huggingface-hub tokenizers
pip install httpx  # For async HTTP requests

# Verify TensorRT-LLM installation
echo "🔍 Verifying TensorRT-LLM installation..."
python -c "
try:
    import tensorrt_llm
    print(f'✅ TensorRT-LLM version: {tensorrt_llm.__version__}')
    
    from tensorrt_llm.runtime import ModelRunner
    print('✅ TensorRT-LLM Runtime available')
    
    from tensorrt_llm import Mapping
    print('✅ TensorRT-LLM Mapping available')
    
except ImportError as e:
    print(f'❌ TensorRT-LLM import failed: {e}')
    exit(1)
"

# Create directories for models and engines
echo "📁 Creating model directories..."
mkdir -p /home/nvidia/models/engines
mkdir -p /home/nvidia/models/tokenizers
mkdir -p /home/nvidia/models/checkpoints

# Download a base tokenizer for testing
echo "📦 Downloading base tokenizer..."
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
tokenizer.save_pretrained('/home/nvidia/models/tokenizers/base-tokenizer')
print('✅ Base tokenizer saved to /home/nvidia/models/tokenizers/base-tokenizer')
"

# Create TensorRT-LLM test script
echo "📝 Creating TensorRT-LLM test script..."
cat > /home/nvidia/test_tensorrt_llm.py << 'EOF'
#!/usr/bin/env python3
"""
TensorRT-LLM Test Script for JustNews V4
Tests TensorRT-LLM installation and basic functionality
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tensorrt_llm():
    """Test TensorRT-LLM installation and components."""
    print("🧪 Testing TensorRT-LLM Installation")
    print("=" * 50)
    
    # Test 1: Import TensorRT-LLM
    try:
        import tensorrt_llm
        print(f"✅ TensorRT-LLM {tensorrt_llm.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TensorRT-LLM: {e}")
        return False
    
    # Test 2: Import runtime components
    try:
        from tensorrt_llm.runtime import ModelRunner, GenerationSession
        print("✅ TensorRT-LLM Runtime components imported")
    except ImportError as e:
        print(f"❌ Failed to import runtime components: {e}")
        return False
    
    # Test 3: Import mapping
    try:
        from tensorrt_llm import Mapping
        print("✅ TensorRT-LLM Mapping imported")
    except ImportError as e:
        print(f"❌ Failed to import Mapping: {e}")
        return False
    
    # Test 4: Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {device_count} device(s)")
            print(f"   Primary device: {device_name}")
        else:
            print("❌ CUDA not available")
            return False
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        return False
    
    # Test 5: Check tokenizer availability
    tokenizer_path = Path("/home/nvidia/models/tokenizers/base-tokenizer")
    if tokenizer_path.exists():
        print(f"✅ Base tokenizer available at {tokenizer_path}")
    else:
        print(f"❌ Base tokenizer not found at {tokenizer_path}")
        return False
    
    # Test 6: Model directories
    dirs_to_check = [
        "/home/nvidia/models/engines",
        "/home/nvidia/models/checkpoints"
    ]
    
    for dir_path in dirs_to_check:
        if Path(dir_path).exists():
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
            return False
    
    print("\n🎉 All TensorRT-LLM tests passed!")
    print("Ready for RTX AI Toolkit model conversion and deployment")
    return True

if __name__ == "__main__":
    success = test_tensorrt_llm()
    sys.exit(0 if success else 1)
EOF

chmod +x /home/nvidia/test_tensorrt_llm.py

# Run the TensorRT-LLM test
echo "🧪 Running TensorRT-LLM test..."
python /home/nvidia/test_tensorrt_llm.py

# Create environment configuration file
echo "📝 Creating environment configuration..."
cat > /home/nvidia/.env.tensorrt << 'EOF'
# TensorRT-LLM Environment Configuration for JustNews V4
TENSORRT_ENGINE_DIR=/home/nvidia/models/engines
TOKENIZER_DIR=/home/nvidia/models/tokenizers/base-tokenizer
RTX_BATCH_SIZE=4
RTX_MAX_TOKENS=512
QUANTIZATION=int4
RTX_PRIMARY_ENDPOINT=tensorrt://localhost:8080
DOCKER_FALLBACK_ENDPOINT=http://model-runner:12434/v1/
AIM_SDK_ENABLED=false
EOF

echo "✅ TensorRT-LLM environment configuration saved to /home/nvidia/.env.tensorrt"

# Create model conversion script template
echo "📝 Creating model conversion script template..."
cat > /home/nvidia/convert_model_to_tensorrt.py << 'EOF'
#!/usr/bin/env python3
"""
Model Conversion Script for JustNews V4
Converts HuggingFace models to TensorRT-LLM engines

Usage:
    python convert_model_to_tensorrt.py --model_dir ./path/to/model --output_dir ./engines/model_name
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_model(model_dir: str, output_dir: str, quantization: str = "int4"):
    """
    Convert HuggingFace model to TensorRT-LLM engine.
    
    This follows the RTX AI Toolkit workflow:
    1. Load HuggingFace checkpoint
    2. Apply quantization (INT4_AWQ)
    3. Build TensorRT-LLM engine
    4. Save optimized engine
    """
    
    print(f"🔧 Converting model from {model_dir} to {output_dir}")
    print(f"Quantization: {quantization}")
    
    try:
        # Import TensorRT-LLM build tools
        from tensorrt_llm.models import LLaMAForCausalLM
        from tensorrt_llm.quantization import QuantMode
        
        # This is a template - actual implementation depends on model type
        logger.info("Model conversion requires specific implementation per model type")
        logger.info("Refer to RTX AI Toolkit documentation for your specific model")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("✅ Model conversion template ready")
        print("📋 Next steps:")
        print("1. Follow RTX AI Toolkit model-specific conversion guide")
        print("2. Use AI Workbench for guided conversion process")
        print("3. Test converted engine with JustNews V4")
        
    except ImportError as e:
        logger.error(f"TensorRT-LLM build tools not available: {e}")
        logger.info("This is expected - use RTX AI Toolkit workflow instead")

def main():
    parser = argparse.ArgumentParser(description="Convert model to TensorRT-LLM")
    parser.add_argument("--model_dir", required=True, help="Path to HuggingFace model")
    parser.add_argument("--output_dir", required=True, help="Output directory for TensorRT engine")
    parser.add_argument("--quantization", default="int4", help="Quantization method")
    
    args = parser.parse_args()
    convert_model(args.model_dir, args.output_dir, args.quantization)

if __name__ == "__main__":
    main()
EOF

chmod +x /home/nvidia/convert_model_to_tensorrt.py

echo ""
echo "🎉 TensorRT-LLM setup complete!"
echo "=================================="
echo ""
echo "✅ TensorRT-LLM installed in RAPIDS environment"
echo "✅ Model directories created"
echo "✅ Base tokenizer downloaded"
echo "✅ Test script created"
echo "✅ Environment configuration ready"
echo "✅ Model conversion template ready"
echo ""
echo "📋 Next Steps:"
echo "1. Run: python /home/nvidia/test_tensorrt_llm.py"
echo "2. Use RTX AI Toolkit to fine-tune and convert models"
echo "3. Update JustNews V4 rtx_manager.py configuration"
echo "4. Test with actual model engines"
echo ""
echo "🔧 Configuration file: /home/nvidia/.env.tensorrt"
echo "📁 Models directory: /home/nvidia/models/"
echo "🧪 Test script: /home/nvidia/test_tensorrt_llm.py"
echo ""
