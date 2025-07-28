# Python Version Compatibility Research
# Determining optimal Python version for JustNews V4 native deployment

## Critical Packages We Need

### Core ML/GPU Stack
- **PyTorch**: GPU acceleration, transformers support
- **Transformers**: HuggingFace models for sentiment/bias analysis  
- **TensorRT-LLM**: NVIDIA inference optimization
- **NVIDIA RAPIDS**: cuDF, cuML, cuGraph for 150x pandas speedup
- **CUDA Toolkit**: GPU compute support

### Web Framework
- **FastAPI**: Web API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### Supporting Libraries
- **sentence-transformers**: Text embeddings
- **accelerate**: HuggingFace GPU acceleration
- **safetensors**: Model serialization

## Compatibility Matrix (Research Needed)

### Python 3.13.2 (Current System Default)
- PyTorch: ❓ Need to check
- RAPIDS: ❓ Need to check  
- TensorRT-LLM: ❓ Need to check
- Transformers: ❓ Need to check

### Python 3.12.x (Likely Best Option)
- PyTorch: ✅ Fully supported
- RAPIDS: ❓ Need to check
- TensorRT-LLM: ❓ Need to check
- Transformers: ✅ Fully supported

### Python 3.11.x (Docker Default)
- PyTorch: ✅ Fully supported
- RAPIDS: ✅ Known to work
- TensorRT-LLM: ✅ Likely supported
- Transformers: ✅ Fully supported

## Research Commands to Run

```bash
# Check RAPIDS Python support
curl -s https://docs.rapids.ai/install | grep -i python

# Check TensorRT-LLM Python support  
curl -s https://pypi.nvidia.com/tensorrt-llm/ | grep -i python

# Check PyTorch Python support
curl -s https://pytorch.org/get-started/locally/ | grep -i python
```

## Environment Cleanup Plan

### Current Environments to Remove
- `justnews-v4` (Python 3.11)
- `justnews-native` (Python 3.13.2)
- Any other conda environments created

### Clean Slate Process
1. List all conda environments: `conda env list`
2. Remove all JustNews environments: `conda env remove -n <env_name>`
3. Verify system Python version: `python3 --version`
4. Research optimal Python version based on package compatibility
5. Create single, optimized conda environment with chosen Python version
6. Install all packages in correct order with proper dependencies

## Recommended Action Plan

1. **Research Phase** (this document)
2. **Environment Cleanup** - Remove all virtual environments
3. **Version Selection** - Choose optimal Python version based on research
4. **Single Environment Creation** - One conda environment for entire system
5. **Native Deployment** - All agents run natively with shared environment
6. **Performance Validation** - Measure actual performance improvements

## Decision Criteria

The chosen Python version must support:
- ✅ Latest PyTorch with CUDA 12.1+ support
- ✅ NVIDIA RAPIDS (cuDF, cuML, cuGraph)
- ✅ TensorRT-LLM for inference optimization
- ✅ All web framework and supporting libraries
- ✅ Stable, production-ready packages (not bleeding edge)

Based on experience and stability, **Python 3.12.x is likely the best choice**.
