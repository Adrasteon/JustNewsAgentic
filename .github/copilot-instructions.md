# JustNews V4 AI Coding Agent Instructions

## System Architecture
JustNews V4 is a multi-agent news analysis system with **native TensorRT GPU acceleration** and **MCP (Model Context Protocol) bus communication**. The architecture consists of 8 specialized agents communicating through a central message bus.

**Current Status**: Native TensorRT Production Deployment - validated with ultra-safe testing achieving **2.69x performance improvement** over baseline with zero crashes and completely clean operation.

### Core Components
- **MCP Bus** (Port 8000): Central communication hub using FastAPI with `/register`, `/call`, `/agents` endpoints
- **Agents** (Ports 8001-8007): Independent FastAPI services with native TensorRT acceleration
- **Database**: PostgreSQL + vector search for semantic article storage
- **GPU Stack**: Water-cooled RTX 3090 with native TensorRT 10.10.0.31, PyCUDA, professional CUDA management

## Native TensorRT Performance (PRODUCTION STRESS TESTED)
**Water-Cooled RTX 3090 Results** - Validated with 1,000 articles × 2,000 characters each:
- **Sentiment Analysis**: **720.8 articles/sec** (production stress tested)
- **Bias Analysis**: **740.3 articles/sec** (production stress tested)  
- **Combined Average**: **730+ articles/sec** sustained throughput
- **Memory Efficiency**: 2.3GB GPU utilization (highly optimized)
- **System Stability**: Zero crashes, zero warnings, 100% reliability
- **Production Validated**: Successfully processed 2M+ characters in 2.7 seconds

## Optimized V4 Architecture (Strategic Pipeline Design)
**Intelligence-First Design** - Scout pre-filtering enables downstream optimization:
- **Scout Agent**: 8GB (LLaMA-3-8B + self-learning) - Critical content quality pre-filter
- **Fact Checker**: 2.5GB (DialoGPT-medium) - Optimized due to Scout pre-filtering clean input
- **Pipeline Efficiency**: Scout's ML-based news detection reduces downstream model requirements
- **Memory Buffer Target**: 2-3GB minimum for production stability and context management

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

## Enhanced Scout Agent Integration (Production Ready)

### Scout Agent Enhanced Deep Crawl System
**Status**: ✅ Production deployment complete with native Crawl4AI integration

**Key Features**:
- **Native Crawl4AI Integration**: Version 0.7.2 with BestFirstCrawlingStrategy
- **Scout Intelligence Engine**: LLaMA-3-8B GPU-accelerated content analysis
- **User-Configurable Parameters**: max_depth=3, max_pages=100, word_count_threshold=500
- **Quality Filtering**: Dynamic threshold-based content selection
- **MCP Bus Communication**: Full integration with agent registration system

**Implementation Pattern**:
```python
# Enhanced deep crawl with Scout Intelligence
async def enhanced_deep_crawl_site(
    url: str,
    max_depth: int = 3,
    max_pages: int = 100,
    word_count_threshold: int = 500,
    quality_threshold: float = 0.6,
    analyze_content: bool = True
) -> List[Dict]:
    # Native Crawl4AI with BestFirstCrawlingStrategy
    strategy = BestFirstCrawlingStrategy(
        max_depth=max_depth,
        max_pages=max_pages,
        filter_chain=FilterChain([
            ContentTypeFilter(["text/html"]),
            DomainFilter(allowed_domains=[domain])
        ]),
        word_count_threshold=word_count_threshold
    )
    
    # Scout Intelligence analysis with quality filtering
    if intelligence_available and scout_engine and analyze_content:
        analysis = scout_engine.comprehensive_content_analysis(content, url)
        scout_score = analysis.get("scout_score", 0.0)
        
        if scout_score >= quality_threshold:
            result["scout_analysis"] = analysis
            result["scout_score"] = scout_score
            # Additional quality metrics and filtering
```

**Performance Validation**:
- Sky News crawl: 148k characters in 1.3 seconds
- Scout Intelligence analysis: Content scoring and quality filtering operational
- MCP Bus integration: Full agent registration and communication validated

## Development Workflows

### Production Environment (VALIDATED)
**Conda Environment**: `rapids-25.06` (Python 3.12)
**Key Packages**: PyTorch 2.2.0+cu121, transformers 4.39.0, sentence-transformers 2.6.1
**Compatibility Fix**: NumPy 1.26.4 (resolved torchvision::nms conflicts)
**Hardware**: Water-cooled RTX 3090 (25.3GB memory available)

### Starting the System
```bash
# Full system (Docker Compose)
docker-compose up --build

# Native GPU analyst (Ubuntu native - PRODUCTION READY)
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
The `native_tensorrt_engine.py` demonstrates the current **native TensorRT production implementation** with professional CUDA management:

#### Current GPU Status (✅ NATIVE TENSORRT PRODUCTION DEPLOYED):
- ✅ **Analyst**: **Native TensorRT production deployment** (730+ articles/sec - 4.8x improvement)
  - ✅ Sentiment Analysis: 720.8 articles/sec (production stress tested)
  - ✅ Bias Analysis: 740.3 articles/sec (production stress tested)
  - ✅ Memory Efficiency: 2.3GB GPU utilization (65% reduction)
  - ✅ Stability: Zero crashes, zero warnings, 100% reliability
  - ✅ Stress Testing: 1,000 articles × 2,000 chars processed successfully
  - ✅ Context Management: Professional CUDA context lifecycle with persistent global engine
- ⏳ **Scout**: LLaMA-3-8B ready for TensorRT migration (8GB - critical content pre-filter)
- ⏳ **Fact Checker**: DialoGPT-medium ready for TensorRT migration (2.5GB - Scout-optimized)
- ⏳ **Synthesizer**: DialoGPT-medium + embeddings ready for TensorRT migration (3GB)
- ⏳ **Critic**: DialoGPT-medium ready for TensorRT migration (2.5GB)  
- ⏳ **Chief Editor**: DialoGPT-medium orchestration focus (2GB - specification-optimized)
- ⏳ **Memory**: Vector embeddings ready for TensorRT migration (1.5GB)

#### Native TensorRT Implementation Pattern (✅ PRODUCTION READY):
```python
# Production-Ready Native TensorRT with Professional CUDA Management
class NativeTensorRTInferenceEngine:
    def __init__(self, engines_dir="tensorrt_engines"):
        # Professional CUDA context management
        self.cuda_context = None
        self.context_created = False
        self._initialize_cuda_context()
        self._load_tensorrt_engines()
    
    def score_sentiment_batch_native(self, texts: List[str]) -> List[Optional[float]]:
        # Native TensorRT execution with FP16 precision
        # 786.8 articles/sec performance validated
        return self._run_native_batch_inference(texts, "sentiment")
    
    def cleanup(self):
        # Proper CUDA context cleanup with Context.pop()
        if self.context_created and self.cuda_context:
            self.cuda_context.pop()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
```
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

#### GPU Memory Allocation Strategy (Native TensorRT):
- **RTX 3090 24GB**: Using only 2.3GB for analyst agent (highly efficient)
- **Native TensorRT**: FP16 precision with compiled engines
- **Batch Processing**: 100-article batches for maximum throughput

### Model Loading Convention
```python
# Native TensorRT engine loading pattern
from native_tensorrt_engine import NativeTensorRTInferenceEngine

# Production-ready TensorRT loading
def load_tensorrt_engine(engines_dir: str = "tensorrt_engines"):
    with NativeTensorRTInferenceEngine(engines_dir=engines_dir) as engine:
        logger.info("✅ Native TensorRT engines loaded")
        return engine

# Fallback to HuggingFace for non-TensorRT agents
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
- `agents/analyst/native_tensorrt_engine.py`: Native TensorRT implementation (PRODUCTION READY)
- `agents/analyst/ultra_safe_tensorrt_test.py`: Production validation testing
- `agents/analyst/tensorrt_engines/`: Compiled TensorRT engines and metadata
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
- **Native TensorRT Performance**: 730+ articles/sec (RTX 3090, production stress tested)
- **Individual Engine Performance**: Sentiment 720.8 art/sec, Bias 740.3 art/sec
- **Memory Efficiency**: 2.3GB GPU utilization (highly optimized)
- **System Stability**: Zero crashes, zero warnings, completely clean operation
- **Production Scale**: 1,000 articles × 2,000 chars processed in 2.7 seconds
- **Target Full GPU Integration**: 800-1000 articles/sec across all agents with TensorRT
- **Docker CPU Baseline**: 0.24-0.6 articles/sec
- **Batch Processing**: 100-article batches for maximum throughput
- **Production Ready**: Ubuntu native deployment with professional CUDA management

## GPU Integration Roadmap (Production TensorRT Achieved)
### Current Status: Native TensorRT Production Success
**Achieved**: 406.9 articles/sec with native TensorRT engines (2.69x improvement over baseline)
**Validated**: Zero crashes, zero warnings, completely clean operation
**Production Ready**: Ultra-safe testing confirms deployment readiness

### Phase 1: Expand TensorRT to All Agents (Current Phase)
1. **Apply TensorRT Pattern**: Migrate remaining agents to native TensorRT implementation
2. **Optimize Engine Loading**: Implement efficient TensorRT engine management across agents
3. **Memory Management**: Professional CUDA context management for multi-agent deployment
4. **Production Validation**: Apply stress testing to validate each agent at production scale
5. **Space-Saving Optimizations**: Target 2-3GB memory buffer for production stability

**Optimized System Memory Allocation (RTX 3090 24GB):**
- Analyst: 2.3GB (✅ TensorRT validated)
- Scout: 8.0GB (LLaMA-3-8B + self-learning - critical pre-filter)
- Fact Checker: 2.5GB (DialoGPT-medium - Scout-optimized)
- Synthesizer: 3.0GB (DialoGPT-medium + embeddings)
- Critic: 2.5GB (DialoGPT-medium)
- Chief Editor: 2.0GB (DialoGPT-medium - orchestration focus)
- Memory: 1.5GB (Vector embeddings)
- **Total: 21.8GB | Buffer: 0.2GB** ⚠️ (Requires additional optimization)

**Target Optimizations for 2-3GB Buffer:**
- Model quantization (INT8 where possible)
- Context window optimization
- Batch size tuning
- Memory-efficient attention mechanisms

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
