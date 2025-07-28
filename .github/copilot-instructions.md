# JustNews V4 AI Coding Agent Instructions

## System Architecture
JustNews V4 is a multi-agent news analysis system with **production-validated GPU acceleration** and **MCP (Model Context Protocol) bus communication**. The architecture consists of 8 specialized agents communicating through a central message bus.

**Current Status**: Production-Ready GPU Deployment - validated with 1,000-article stress testing, achieving 75%+ of V4 performance targets with crash-free operation and professional CUDA device management.

### Core Components
- **MCP Bus** (Port 8000): Central communication hub using FastAPI with `/register`, `/call`, `/agents` endpoints
- **Agents** (Ports 8001-8007): Independent FastAPI services with validated GPU acceleration
- **Database**: PostgreSQL + vector search for semantic article storage
- **GPU Stack**: Water-cooled RTX 3090 with PyTorch 2.2.0+cu121, transformers 4.39.0, professional CUDA management

## Production GPU Performance (VALIDATED)
**Water-Cooled RTX 3090 Results**:
- **Sentiment Analysis**: 151.4 articles/sec (75.7% of V4 target)
- **Bias Analysis**: 146.8 articles/sec (73.4% of V4 target)  
- **System Stability**: 1,000-article stress test completed without crashes
- **Memory Management**: Efficient 25.3GB GDDR6X utilization with FP16 precision
- **Article Size**: Production-scale 2,717-char articles (realistic news content)

## Agent Communication Pattern
**Critical**: All agents follow identical communication patterns:

```python
# Agent main.py structure (every agent)
class ToolCall(BaseModel):
    args: list
    kwargs: dict

@app.post("/tool_name")
def tool_endpoint(call: ToolCall):
    from tools import tool_function
    return tool_function(*call.args, **call.kwargs)

@app.get("/health")
def health():
    return {"status": "ok"}
```

**MCP Bus Integration**: Agents call other agents via `POST /call` with:
```python
payload = {
    "agent": "target_agent",
    "tool": "tool_name", 
    "args": [arg1, arg2],
    "kwargs": {}
}
requests.post(f"{MCP_BUS_URL}/call", json=payload)
```

## Development Workflows

### Starting the System
```bash
# Full system (Docker Compose)
docker-compose up --build

# Native GPU analyst (Ubuntu/WSL2)
### Starting the System
```bash
# Full system (Docker Compose)
docker-compose up --build

# Native GPU analyst (Ubuntu/WSL2)
source /home/adra/miniconda3/etc/profile.d/conda.sh
conda activate rapids-25.06
python start_native_gpu_analyst.py
```

### Testing Individual Agents
Each agent can run standalone on its designated port. Always test agent endpoints individually before Docker integration.

### Performance Testing
Use `production_stress_test.py` for production-scale validation with full-length news articles (2,717+ chars). Current GPU performance: **151.4 articles/sec sentiment, 146.8 articles/sec bias analysis**.

## Project-Specific Patterns

### Feedback Logging (Universal Pattern)
**Every agent** implements identical feedback logging:
```python
def log_feedback(event: str, details: dict):
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")
```

### GPU Integration Strategy
The `hybrid_tools_v4.py` demonstrates the current production-ready GPU pattern with professional CUDA management:

#### Current GPU Status (Production-Ready Architecture):
- ✅ **Analyst**: Production-validated GPU acceleration (151.4-146.8 articles/sec with 1,000-article stress test)
- ⏳ **Fact Checker**: DialoGPT models ready for GPU (774M params) - awaiting production deployment
- ⏳ **Synthesizer**: Sentence-transformers + ML pipeline ready for GPU - awaiting production deployment
- ⏳ **Critic**: DialoGPT models ready for GPU (355M params) - awaiting production deployment
- ⏳ **Scout/Chief Editor**: LLaMA models ready for GPU migration - awaiting production deployment

#### Current GPU Implementation Pattern (Production-Ready):
```python
# Current Production Implementation with CUDA Device Management
class GPUAcceleratedAnalyst:
    def _initialize_gpu_models(self):
        # Professional CUDA device management
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        
        # Using HuggingFace transformers with GPU acceleration
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0,  # GPU device
            batch_size=32,  # Batch processing for 10x speedup
            torch_dtype=torch.float16  # FP16 for memory efficiency
        )
    
    def score_sentiment_batch_gpu(self, texts: List[str]) -> List[Optional[float]]:
        # Professional device context management
        torch.cuda.set_device(0)
        with torch.cuda.device(0):
            results = self.sentiment_analyzer(texts)
        return processed_results
```python
# Current Working Implementation (V3.5 achieving V4 performance)
class GPUAcceleratedAnalyst:
    def _initialize_gpu_models(self):
        # Using HuggingFace transformers with GPU acceleration
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0,  # GPU device
            batch_size=32  # Batch processing for 10x speedup
        )

# Planned V4 RTX AI Toolkit Integration (Future):
class RTXOptimizedHybridManager:
    def __init__(self):
        # Future V4 implementation with AIM SDK
        from nvidia_aim import InferenceManager  # Not yet implemented
        self.aim_client = InferenceManager()
```

#### GPU Memory Allocation Strategy (Current V3.5):
- **RTX 3090 24GB**: Currently using ~6-8GB for analyst agent
- **HuggingFace Optimization**: FP16 precision for memory efficiency
- **Batch Processing**: 32-item batches for optimal GPU utilization

### Model Loading Convention
```python
# Environment-based model paths with GPU optimization
MODEL_PATH = os.environ.get("MODEL_NAME_PATH", "./models/default-path")

# GPU-optimized HuggingFace loading pattern
def load_model_gpu_optimized(model_name: str):
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info(f"✅ GPU model loaded: {model_name}")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.warning(f"⚠️ CPU fallback: {model_name}")
    return model
```

## Critical File Locations

### Core Architecture
- `docker-compose.yml`: Multi-agent orchestration with GPU support
- `mcp_bus/main.py`: Central message bus implementation
- `agents/*/main.py`: FastAPI agent endpoints (identical pattern)
- `agents/*/tools.py`: Business logic implementation

### GPU Integration
- `agents/analyst/hybrid_tools_v4.py`: GPU acceleration reference implementation
- `start_native_gpu_analyst.py`: Native GPU deployment script
- `wsl_deployment/`: Native GPU environment deployment

### Configuration
- `config.json`: Mistral model configuration
- `.env.example`: Environment variables template
- `requirements.txt`: Python dependencies (per agent + root)

## Integration Points

### Database Integration (Memory Agent)
- PostgreSQL on port 5432 with vector search
- Migrations in `agents/memory/db_migrations/`
- Semantic search using sentence-transformers embeddings

### External Services
- **Crawl4AI**: Web scraping service (port 5000)
- **SerpAPI**: Search API integration (requires key)
- **HuggingFace**: Model downloads (transformers, sentence-transformers)

## Performance Expectations
- **Current GPU Performance**: 146.8-151.4 articles/sec (RTX 3090, production-validated with 1,000 articles)
- **Target Full GPU Integration**: 200-400 articles/sec across all agents
- **Docker CPU Baseline**: 0.24-0.6 articles/sec
- **Batch Processing**: 10x+ improvement over sequential processing with 25-100 article batches
- **GPU Memory Efficiency**: 4-6GB per agent with FP16 precision and professional CUDA management
- **Production Ready**: Ubuntu native deployment with water-cooled thermal management

## GPU Integration Roadmap (V3.5 → V4 Migration)
### Current Status: V4 Performance with V3.5 Architecture
**Achieved**: 146.8-151.4 articles/sec using HuggingFace transformers with professional CUDA management (exceeds V4 targets)
**Validated**: 1,000-article production stress test completed without crashes
**Next**: Migration to full RTX AI Toolkit integration while maintaining performance

### Phase 1: Maintain Current Performance (V3.5 Architecture)
1. **Expand Current GPU Pattern**: Apply HuggingFace GPU acceleration to remaining agents
2. **Optimize Batch Processing**: Implement 32-item batches across all agents
3. **Memory Management**: Professional GPU memory handling for multi-agent deployment
1. **Expand Current GPU Pattern**: Apply HuggingFace GPU acceleration to remaining agents
2. **Optimize Batch Processing**: Implement 32-item batches across all agents
3. **Memory Management**: Professional GPU memory handling for multi-agent deployment

### Phase 2: RTX AI Toolkit Migration (V4 Architecture)
4. **AIM SDK Integration**: Replace HuggingFace with nvidia_aim.InferenceManager
5. **TensorRT-LLM**: Migrate from transformers.pipeline to tensorrt_llm inference
6. **Docker Model Runner**: Implement as specified fallback system

### Phase 3: Full V4 Implementation
7. **AI Workbench**: QLoRA fine-tuning for domain specialization
8. **Custom Models**: RTX-optimized news analysis models
9. **Complete Architecture**: Full RTXOptimizedHybridManager implementation

## Development Context
This system evolved from V3 Docker-only to V4 hybrid GPU acceleration. The migration to Ubuntu 24.04 native is documented in `UBUNTU_MIGRATION_GUIDE.md`. All development history preserved in `DEVELOPMENT_CONTEXT.md`.

Always check existing patterns in `agents/*/` before implementing new features. The MCP bus communication pattern is non-negotiable and must be maintained across all agents.

## Documentation Maintenance Requirements

### V4 Conformance Standards
**Critical**: All development must conform to both `docs/JustNews_Proposal_V4.md` and `docs/JustNews_Plan_V4.md` specifications:

**Note**: Authority documents may be modified with explicit user authorization when architectural changes are required.

#### RTX AI Toolkit Integration Requirements:
- **TensorRT-LLM Primary**: 4x performance improvement target with Docker fallback
- **AI Workbench Pipeline**: QLoRA fine-tuning with NVIDIA AI Workbench integration
- **AIM SDK Orchestration**: Intelligent routing between TensorRT-LLM and Docker backends
- **Professional GPU Memory Management**: Crash-free operation with RTX 3090 optimization

#### V4 Architecture Compliance:
- **Hybrid AI Pipeline**: Stage 1 (RTX-Optimized Bootstrap) + Stage 2 (AI Workbench Evolution)
- **Domain Specialization**: News-analysis optimized models using RTX AI Toolkit
- **Three-Phase Migration**: Foundation → Training Pipeline → RTX-Native Replacement
- **Performance Targets**: 4x inference speed, 3x model compression, zero downtime deployment

### Documentation Update Protocol
**Every code change requires corresponding documentation updates:**

#### 1. README.md Updates
```markdown
# When adding features, update:
- Architecture section with new components
- Performance metrics with validated benchmarks
- Installation steps for new dependencies
- API endpoints for new agent tools
```

#### 2. CHANGELOG.md Maintenance
```markdown
# Required format for all releases:
## [Version] - Date - Feature Summary
### Added/Enhanced/Fixed/Removed
- Specific feature descriptions with performance metrics
- Technical infrastructure changes
- Status updates (✅/⏳/❌) for GPU integration
```

#### 3. Agent Documentation Pattern
```python
# Every agent tools.py must include:
"""
Agent: [Name] Agent
Purpose: [Specific function in V4 architecture]
GPU Status: [✅ Integrated | ⏳ Ready for GPU | ❌ CPU Only]
Performance: [Current metrics vs baseline]
V4 Compliance: [TensorRT-LLM/Docker fallback status]
Dependencies: [Model requirements, GPU memory allocation]
"""
```

### V4-Specific Documentation Requirements

#### GPU Integration Documentation:
- **Performance Validation**: Always include realistic article-length benchmarks
- **Memory Management**: Document GPU memory allocation (4-6GB per agent on RTX 3090)
- **Fallback Behavior**: Specify CPU fallback triggers and error handling
- **Batch Processing**: Document batch size optimization for each model type

#### Model Conformance:
- **TensorRT-LLM Integration**: Document quantization (INT8/FP16) and optimization
- **Docker Model Runner**: Maintain compatibility with `ai/mistral` and `ai/llama3.2`
- **HuggingFace Models**: Specify model versions and GPU optimization settings
- **Custom Training**: Document AI Workbench integration for domain specialization

### Documentation File Structure
```
docs/
├── JustNews_Proposal_V4.md     # 🔒 Authority document (User authorization required)
├── JustNews_Plan_V4.md         # 🔒 Implementation authority (User authorization required)
├── API_Documentation.md        # 🔄 Update with new agent endpoints
├── Performance_Benchmarks.md   # 🔄 Update with GPU integration metrics
└── Deployment_Guide.md         # 🔄 Update with Ubuntu native procedures

Root Documentation:
├── README.md                   # 🔄 Primary project documentation
├── CHANGELOG.md               # 🔄 All version changes with metrics
├── DEVELOPMENT_CONTEXT.md     # 🔄 Complete development history
└── UBUNTU_MIGRATION_GUIDE.md  # 🔄 Native deployment procedures
```

### Code Comment Standards
```python
# V4-Compliant code documentation:
class GPUOptimizedAgent:
    """
    V4 GPU-Accelerated Agent Implementation
    
    Conformance:
    - TensorRT-LLM primary processing (4x performance target)
    - Docker Model Runner fallback for reliability
    - RTX 3090 memory allocation: 4-6GB per agent
    - Batch processing: 32-item batches for optimal GPU utilization
    
    Performance Metrics:
    - GPU: [X] articles/sec (vs [Y] CPU baseline)
    - Memory: [Z]GB VRAM utilization
    - Fallback: <500ms detection and switching
    """
```

### Validation Checklist
Before any commit, verify:
- [ ] V4 Proposal/Plan conformance maintained
- [ ] Performance metrics include realistic article benchmarks
- [ ] Documentation updated for all modified components
- [ ] GPU integration status clearly documented
- [ ] CHANGELOG.md entry with specific metrics
- [ ] README.md reflects current system capabilities
