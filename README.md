# JustNewsAgentic V4

This project implements the JustNews V4 system, an agentic, MCP-first news analysis ecosystem with **NVIDIA RTX AI Toolkit int## 🚀 Ubuntu Migration Readiness

### Migration Status: Complete Preparation ✅

All necessary files and documentation have been prepared for Ubuntu 24.04 dual-boot migration:

- **UBUNTU_MIGRATION_GUIDE.md**: Complete 7-phase migration process
- **prepare-ubuntu-migration.ps1**: Automated backup and git preparation
- **verify-ubuntu-migration.sh**: Post-migration verification and validation
- **DEVELOPMENT_CONTEXT.md**: Comprehensive project history and context preservation

### Expected Performance Improvements
- **Current WSL2**: 5.7 articles/sec GPU batch processing
- **Ubuntu Native**: 8-12 articles/sec expected (40-110% improvement)
- **Reason**: Direct GPU access without Windows/WSL2 virtualization overhead

### Migration Command Summary
```powershell
# 1. Backup everything
.\prepare-ubuntu-migration.ps1

# 2. Commit and branch
git add -A && git commit -m "Pre-Ubuntu migration: V4 hybrid ready"
git push origin main
git checkout -b justnews-v4-ubuntu

# 3. Proceed with dual-boot setup (see UBUNTU_MIGRATION_GUIDE.md)
```

## 🔧 Development Environments designed as a collaborative group of specialized AI agents that work together to find, analyze, and synthesize news stories with professional-grade GPU acceleration.

## 🚀 V4 Current Status: Production Ready with Ubuntu Migration Prepared

**Latest Achievement**: Complete GPU acceleration implementation with honest performance metrics (July 27, 2025)

### ✅ Successfully Implemented
- **GPU-Accelerated Analyst**: 5.7 articles/sec batch processing (24.2x faster than CPU)
- **TensorRT-LLM 0.20.0**: Fully operational on RTX 3090 with 24GB VRAM
- **RAPIDS 25.6.0**: GPU-accelerated data science suite with proven 2.8x speedup
- **Batch Processing**: 10.3x improvement over sequential processing (0.6→5.7 articles/sec)
- **Realistic Performance Metrics**: Validated with 1,200+ character news articles
- **Ubuntu Migration Ready**: Complete migration guide and automation scripts prepared

### 📊 Validated Performance (Real Articles)
- **GPU Batch Processing**: 5.7 articles/sec
- **CPU Baseline**: 0.24 articles/sec  
- **GPU Speedup**: 24.2x faster than CPU
- **Expected Ubuntu Native**: 8-12 articles/sec (40-110% improvement over WSL2)

## Architecture

The system is built on a microservices architecture where each service is an independent AI agent. These agents communicate via a central **MCP (Model Context Protocol) Message Bus**. This allows for a flexible, scalable, and dynamic system where agents can delegate tasks and collaborate to achieve complex goals.

**V4 RTX Architecture**: JustNews V4 introduces GPU-accelerated news analysis using NVIDIA RTX AI Toolkit with TensorRT-LLM for maximum performance, while maintaining Docker Model Runner fallback for reliability. This delivers professional-grade AI performance optimized for RTX 3090.

For full architectural details, see:
- **V4 (Current)**: `docs/JustNews_Proposal_V4.md` and `docs/JustNews_Plan_V4.md`
- **V3 (Legacy)**: `docs/JustNews_Proposal_V3.md` and `docs/JustNews_Plan_V3.md`


## Getting Started

### Prerequisites

- **NVIDIA Hardware**: RTX 3090 (24GB VRAM) or equivalent GPU with CUDA support
- **WSL2 + Ubuntu 24.04**: NVIDIA-SDKM-Ubuntu-24.04 distribution recommended
- **NVIDIA SDK Manager**: For RAPIDS and TensorRT-LLM installation
- **Docker and Docker Compose**: With GPU support (NVIDIA Container Toolkit)
- **Python 3.12**: For RAPIDS environment and TensorRT-LLM
- **RTX AI Toolkit**: Cloned from https://github.com/NVIDIA/RTX-AI-Toolkit

### V4 RTX Environment Setup

The V4 system now runs on a professional GPU-accelerated environment:

1. **RAPIDS Environment (Primary)**:
   ```bash
   # Activate the RAPIDS environment with TensorRT-LLM
   source /home/nvidia/.venvs/rapids25.06_python3.12/bin/activate
   
   # Verify installation
   python -c "import tensorrt_llm; print('✅ TensorRT-LLM Ready!')"
   python -c "import cudf; print('✅ RAPIDS Ready!')"
   ```

2. **Environment Variables**:
   ```bash
   export OMPI_MCA_plm=isolated
   export OMPI_MCA_btl_vader_single_copy_mechanism=none
   export OMPI_MCA_rmaps_base_oversubscribe=1
   export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
   ```

3. **Quick Health Check**:
   ```bash
   # Full system test
   python test_tensorrt_llm.py
   
   # GPU status
   nvidia-smi
   ```

4. **Install NVIDIA Container Toolkit (Windows WSL2 or Linux):**
   ```bash
   # Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   ```

### Build & Run

1. **GPU-Accelerated Environment (WSL2)**:
   ```bash
   # Activate the RAPIDS environment with TensorRT-LLM
   source /home/nvidia/.venvs/rapids25.06_python3.12/bin/activate
   
   # Launch the hybrid V4 system
   cd /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment
   python main.py
   ```

2. **Docker Multi-Agent System (5 agents)**:
    ```bash
    # From Windows PowerShell or WSL
    docker-compose up --build
    ```

3. **System Components**:
    - **GPU-Accelerated (WSL Native)**:
      - `analyst`: GPU batch processing (5.7 articles/sec) - sentiment, bias, entity analysis with TensorRT-LLM/RAPIDS
    
    - **Docker-Based**:
      - `mcp_bus`: Central message bus for agent communication
      - `chief_editor`: Orchestrates news workflows
      - `scout`: Discovers and crawls news sources  
      - `fact_checker`: Validates news and verifies claims
      - `synthesizer`: Clusters and synthesizes articles using ML models
      - `critic`: Reviews synthesis and neutrality using LLM-based critique
      - `memory`: Unified data access (PostgreSQL, vector search)
      - `db`: PostgreSQL database

4. **Performance Expectations**:
   - **Current (WSL2)**: 5.7 articles/sec GPU batch processing
   - **Ubuntu Native**: 8-12 articles/sec expected (40-110% improvement)

### Database Setup & Migrations


Database migrations are located in `agents/memory/db_migrations/`. To apply them manually (from the project root):

```bash
docker-compose exec db psql -U user -d justnews -f /app/db_migrations/001_create_articles_table.sql
docker-compose exec db psql -U user -d justnews -f /app/db_migrations/002_create_training_examples_table.sql
docker-compose exec db psql -U user -d justnews -f /app/db_migrations/003_create_article_vectors_table.sql
```


### ML, Feedback Loops & Continual Learning

All major agents implement machine learning-based feedback loops for continual improvement:

- **Synthesizer**: Uses sentence-transformers for clustering (KMeans, BERTopic, HDBSCAN), LLM for neutralization/aggregation, and logs feedback from Critic/Chief Editor for continual learning. Clustering method is set via `SYNTHESIZER_CLUSTER_METHOD`.
- **Critic**: Uses LLM for critique, logs all feedback and editorial outcomes for retraining and adaptation. Optional fact-checking pipeline can be enabled with `CRITIC_USE_FACTCHECK=1`.
- **Memory**: Provides semantic retrieval with embeddings and vector search, logs all retrievals and downstream outcomes for future learning-to-rank and model improvement. Tool interfaces are mirrored in `tools.py` for clarity and local testing.

**Feedback Logging:**
- All agents log feedback to agent-specific files (e.g., `./feedback_synthesizer.log`) and/or the database.
- Feedback includes tool usage, outcomes, errors, and user/editorial input.
- Standardized logging format: UTC timestamp, event, details (as JSON/dict).

**Retraining & Continual Learning:**
- Feedback logs are used for both online and scheduled retraining.
- Retraining procedures are documented in each agent's code and can be automated via scripts or CI/CD.
- See `action_plan.md` and agent `tools.py` for retraining hooks and feedback usage.

### API Endpoints

Each agent exposes its tools as HTTP endpoints (see `main.py` in each agent folder). The MCP bus provides `/register`, `/call`, and `/agents` endpoints for agent discovery and tool invocation.

### Docker GPU Troubleshooting

**GPU Access Issues:**
- Ensure NVIDIA Container Toolkit is installed and configured
- Verify Docker can access GPU: `docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi`
- Check GPU resources in Docker Desktop settings (increase memory limits)

**Model Loading Issues:**
- Verify model files exist: `ls -la ~/mistral_models/7B-Instruct-v0.3/`
- Check volume mounting in docker-compose logs
- For Windows, copy `docker-compose.override.example.yml` to `docker-compose.override.yml` and adjust paths

**Memory Issues:**
- Analyst agent requires ~14.5GB VRAM for optimal performance
- If OOM errors occur, check available GPU memory: `nvidia-smi`
- Consider reducing batch size or using CPU fallback by removing GPU config

### Dual Functionality: Standalone Execution & MCP Bus Integration

Each agent can be started independently while maintaining the ability to communicate with other agents via the MCP Bus. Follow the instructions below for standalone execution and MCP Bus integration:

#### Standalone Execution
Each agent can operate independently without relying on other agents or services. See the "Standalone Execution" section above for detailed instructions.

#### MCP Bus Integration
When the MCP Bus is available, agents will automatically register their tools and use the bus for inter-agent communication. This allows for dynamic collaboration between agents.

**Key Features:**
- Agents register their tools with the MCP Bus upon startup.
- If the MCP Bus is unavailable, agents will continue to operate independently.
- Error handling ensures smooth operation in standalone mode.

**Starting the MCP Bus:**
1. Navigate to the `mcp_bus` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the MCP Bus:
    ```bash
    uvicorn main:app --reload --port 8000
    ```

**Agent Registration:**
Agents will automatically attempt to register their tools with the MCP Bus at `http://localhost:8000`. Ensure the MCP Bus is running before starting agents for full functionality.

### Standalone Execution

Each agent can be started independently without relying on other agents or services. Follow the instructions below for standalone execution:

#### Chief Editor Agent
1. Navigate to the `agents/chief_editor` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the agent:
    ```bash
    uvicorn main:app --reload --port 8001
    ```

#### Scout Agent
1. Navigate to the `agents/scout` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the agent:
    ```bash
    uvicorn main:app --reload --port 8002
    ```

#### Fact-Checker Agent
1. Navigate to the `agents/fact_checker` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the agent:
    ```bash
    uvicorn main:app --reload --port 8003
    ```

## 📁 Key Files for Migration

### Critical Development Context
- **DEVELOPMENT_CONTEXT.md**: Complete project history and technical decisions
- **V4_INTEGRATION_COMPLETE.md**: Integration completion documentation
- **QUICK_WIN_SUCCESS.md**: Performance validation results
- **real_model_test_results.json**: Honest performance metrics with real articles

### GPU Integration Files
- **agents/analyst/hybrid_tools_v4.py**: GPU-accelerated analyst with batch processing
- **wsl_deployment/**: Native WSL deployment with performance validation
- **quick_win_tensorrt.py**: Performance testing and validation scripts

### Migration Assets
- **UBUNTU_MIGRATION_GUIDE.md**: Complete dual-boot setup guide
- **prepare-ubuntu-migration.ps1**: Automated backup and preparation
- **verify-ubuntu-migration.sh**: Post-migration verification

### System Configuration
- **docker-compose.yml**: Multi-agent orchestration
- **config.json**: System-wide configuration
- **requirements.txt**: Python dependencies

---

## 🎯 Development Status Summary

**V4 Achievement**: Successfully integrated NVIDIA RTX 3090 GPU acceleration with realistic performance validation
- ✅ GPU-accelerated Analyst: TensorRT-LLM + RAPIDS integration
- ✅ Honest Performance Metrics: 5.7 articles/sec (24.2x faster than CPU)
- ✅ Batch Processing: 10.3x improvement over sequential processing
- ✅ Ubuntu Migration Ready: Complete automation and verification scripts
- ⏳ Remaining Work: Full multi-agent GPU integration (5 agents pending)

**Next Phase**: Ubuntu 24.04 native deployment for 40-110% performance improvement
    ```bash
    pip install -r requirements.txt
    ```
3. Start the agent:
    ```bash
    uvicorn main:app --reload --port 8004
    ```

#### Synthesizer Agent
1. Navigate to the `agents/synthesizer` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the agent:
    ```bash
    uvicorn main:app --reload --port 8005
    ```

#### Critic Agent
1. Navigate to the `agents/critic` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the agent:
    ```bash
    uvicorn main:app --reload --port 8006
    ```

#### Memory Agent
1. Navigate to the `agents/memory` directory.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the agent:
    ```bash
    uvicorn main:app --reload --port 8007
    ```