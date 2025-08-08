# Production Deployment Guide - Dependency Management

## Problem Analysis
The dependency conflicts occurred because:
1. Installing spaCy in the RAPIDS environment (rapids-25.06) with strict CUDA dependencies
2. Version conflicts between TensorRT-LLM (requires numpy<2) and spaCy (installed numpy 2.3.2)
3. Multiple RAPIDS packages with incompatible version constraints

## Production Solution

### Option 1: Dedicated Production Environment (Recommended)
```bash
# Create clean environment for production
conda env create -f environment-production.yml
conda activate justnews-production

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Test agents
python test_analyst_v2_specialization.py
python test_critic_v2_specialization.py
```

### Option 2: Docker Production Deployment (Best for deployment)
```bash
# Build production containers with proper dependencies
docker-compose build
docker-compose up

# Each agent container has isolated dependencies
# No conflicts between RAPIDS/TensorRT and spaCy
```

### Option 3: Virtual Environment (Development)
```bash
# Create isolated environment
python -m venv justnews-venv
source justnews-venv/bin/activate

# Install compatible versions
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Key Production Principles Applied

### 1. **No Hard Version Pinning**
```python
# BAD - Creates future conflicts
spacy==3.8.7
numpy==2.3.2

# GOOD - Compatible ranges
spacy>=3.6.0,<4.0.0
numpy>=1.24.0,<2.0.0
```

### 2. **Environment Separation**
- **RAPIDS Environment**: GPU/CUDA intensive tasks
- **Production Environment**: NLP/News analysis tasks  
- **No mixing**: Avoid dependency conflicts

### 3. **Graceful Fallbacks**
```python
# Production-ready fallback handling
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    # Use transformer-based NER fallback
```

### 4. **Docker Isolation**
Each agent runs in isolated container with exact dependencies needed.

## Current Status
- ✅ **Requirements Updated**: Compatible version ranges
- ✅ **Environment Config**: Clean production environment  
- ✅ **Fallback Systems**: Graceful degradation when packages missing
- ✅ **Production Standards**: Zero pinning, proper ranges

## Testing Instructions
```bash
# Test in production environment
conda activate justnews-production
python test_analyst_v2_specialization.py  # Should work without warnings
python test_critic_v2_specialization.py   # Should work without conflicts

# Test fallback behavior (without spaCy)
pip uninstall spacy -y
python test_analyst_v2_specialization.py  # Should fall back to transformers
```

This approach ensures:
- **No version conflicts** 
- **Future compatibility**
- **Production readiness**
- **Graceful degradation**
