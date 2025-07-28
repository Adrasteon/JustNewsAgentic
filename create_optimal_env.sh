#!/bin/bash
# JustNews V4 Optimal Environment Creation

set -e  # Exit on any error

echo "=== Creating JustNews V4 Optimized Environment ==="
echo "Target: Python 3.12 with full GPU acceleration"
echo

# Step 1: Create base environment with Python 3.12
echo "📦 Creating conda environment with Python 3.12..."
conda create -n justnews-v4-optimized python=3.12 -y

# Step 2: Activate environment
echo "🔧 Activating environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate justnews-v4-optimized

# Step 3: Install CUDA toolkit first (foundation)
echo "🚀 Installing CUDA toolkit..."
conda install cuda-toolkit -c nvidia -y

# Step 4: Install PyTorch with CUDA support (critical dependency)
echo "🔥 Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 5: Install RAPIDS (massive performance boost)
echo "⚡ Installing RAPIDS 25.06..."
conda install -c rapidsai -c conda-forge -c nvidia rapids=25.06 python=3.12 cuda-version=12.1 -y

# Step 6: Install TensorRT components
echo "🎯 Installing TensorRT..."
pip install tensorrt --index-url https://pypi.nvidia.com
pip install torch-tensorrt

# Step 7: Install HuggingFace ecosystem
echo "🤗 Installing HuggingFace ecosystem..."
pip install transformers accelerate datasets sentence-transformers

# Step 8: Install FastAPI and web framework
echo "🌐 Installing FastAPI and web components..."
pip install fastapi uvicorn requests aiohttp

# Step 9: Install additional dependencies
echo "📚 Installing additional dependencies..."
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
pip install psycopg2-binary sqlalchemy
pip install python-dotenv pydantic

# Step 10: Verify installation
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import cudf; print(f'RAPIDS cuDF: {cudf.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo
echo "🎉 JustNews V4 Optimized Environment Ready!"
echo "To activate: conda activate justnews-v4-optimized"
echo
echo "Next Steps:"
echo "1. Test native GPU deployment: python start_native_gpu_analyst.py"
echo "2. Run performance benchmarks: python real_model_test.py"
echo "3. Deploy all agents as native Ubuntu services"
