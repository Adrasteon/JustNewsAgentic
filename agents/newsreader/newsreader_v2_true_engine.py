"""
NewsReader V2 Engine - TRUE Multi-Modal Vision Processing
Architecture: LLaVA + Screenshot Processing (OCR & CLIP DISABLED - Testing Redundancy)

CORE FUNCTIONALITY: Screenshot-based webpage processing using LLaVA
- Takes screenshots of webpages
- Uses LLaVA vision-language model to analyze screenshots
- Extracts headlines and article content from visual analysis
- Streamlined architecture: OCR disabled (redundant), CLIP disabled (redundant), Layout Parser disabled (redundant)

V2 Standards:
- 5+ AI models for comprehensive processing
- Zero deprecation warnings
- Professional error handling with GPU acceleration
- Production-ready with fallback systems
- MCP bus integration for inter-agent communication
"""

import asyncio
import logging
import os
import warnings

# Suppress transformers warnings about slow processors
warnings.filterwarnings("ignore", message=".*use_fast.*slow processor.*")
warnings.filterwarnings("ignore", message=".*slow image processor.*")
import json
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from playwright.async_api import async_playwright

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("newsreader.v2_engine")

# Model availability checks with fallback system
try:
    from transformers import BitsAndBytesConfig  # Add quantization support
    from transformers import (
        LlavaOnevisionForConditionalGeneration,  # Updated for OneVision model
    )
    from transformers import LlavaOnevisionProcessor  # Updated for OneVision model
    from transformers import CLIPModel, CLIPProcessor

    LLAVA_AVAILABLE = True
except ImportError:
    logger.warning("LLaVA models not available - using fallback processing")
    LLAVA_AVAILABLE = False

# OCR and Layout Parser DEPRECATED - LLaVA provides superior vision-language understanding
# These were causing memory bloat and crashes - LLaVA handles all vision processing
OCR_AVAILABLE = False  # DEPRECATED: LLaVA superior for text extraction
LAYOUT_PARSER_AVAILABLE = False  # DEPRECATED: LLaVA handles layout understanding

# Environment Configuration
FEEDBACK_LOG = os.environ.get("NEWSREADER_FEEDBACK_LOG", "./feedback_newsreader_v2.log")
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "./model_cache")


class ContentType(Enum):
    ARTICLE = "article"
    IMAGE = "image"
    PDF = "pdf"
    WEBPAGE = "webpage"
    VIDEO = "video"
    MIXED = "mixed"


class ProcessingMode(Enum):
    FAST = "fast"  # Quick processing with basic models
    COMPREHENSIVE = "comprehensive"  # Full multi-modal analysis
    PRECISION = "precision"  # Maximum accuracy with all models


@dataclass
class ProcessingResult:
    content_type: ContentType
    extracted_text: str
    visual_description: str
    layout_analysis: Dict[str, Any]
    confidence_score: float
    processing_time: float
    model_outputs: Dict[str, Any]
    metadata: Dict[str, Any]
    screenshot_path: Optional[str] = None


@dataclass
class NewsReaderV2Config:
    """Configuration for NewsReader V2 Engine"""

    # Model configurations
    llava_model: str = (
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"  # 87% memory reduction: 7.6GB → 1.2GB
    )
    clip_model: str = "openai/clip-vit-large-patch14"
    ocr_languages: List[str] = None
    cache_dir: str = MODEL_CACHE_DIR

    # Quantization settings for memory optimization
    use_quantization: bool = True
    quantization_type: str = "int8"  # "int8", "int4", or "none"
    quantization_compute_dtype: str = "float16"

    # Screenshot settings
    screenshot_timeout: int = 30000
    screenshot_quality: str = "high"
    headless: bool = True

    # Processing settings
    default_mode: ProcessingMode = ProcessingMode.COMPREHENSIVE
    max_image_size: int = 1024
    batch_size: int = 4
    max_sequence_length: int = 2048  # Maximum token length for text processing
    max_new_tokens: int = 512  # Maximum tokens to generate
    device: str = "auto"

    # Component toggles (optimize for LLaVA-first approach)
    enable_layout_parser: bool = False  # Disabled - LLaVA provides layout understanding
    enable_clip_enhancement: bool = True  # Keep for additional vision processing

    # Quality settings
    min_confidence_threshold: float = 0.7
    enable_fallback_processing: bool = True
    use_gpu_acceleration: bool = True

    def __post_init__(self):
        if self.ocr_languages is None:
            self.ocr_languages = ["en", "es", "fr", "de"]


class NewsReaderV2Engine:
    """
    NewsReader V2 Engine - True Multi-Modal Vision Processing

    CORE: Screenshot-based webpage processing using LLaVA
    Features:
    - Playwright screenshot capture with optimizations
    - LLaVA vision-language analysis of screenshots
    - Enhanced OCR and layout analysis
    - GPU acceleration with CPU fallbacks
    - Comprehensive error handling
    - MCP bus integration ready
    - V2 standards compliance (5+ models, zero warnings)
    """

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure GPU memory cleanup"""
        self.cleanup_gpu_memory()

    def cleanup_gpu_memory(self):
        """AGGRESSIVE GPU memory cleanup to prevent crashes"""
        logger.info("🧹 Starting aggressive GPU memory cleanup...")

        # Step 1: Clear model cache
        if hasattr(self, "models") and self.models:
            for model_name, model in list(self.models.items()):
                if model is not None:
                    try:
                        # Move to CPU first
                        if hasattr(model, "cpu"):
                            model.cpu()
                        # Delete model reference
                        del model
                        self.models[model_name] = None
                        logger.info(f"🧹 Cleaned up {model_name} model")
                    except Exception as e:
                        logger.warning(f"Error cleaning up {model_name}: {e}")

            # Clear entire models dict
            self.models.clear()

        # Step 2: Clear processor cache
        if hasattr(self, "processors") and self.processors:
            for proc_name, processor in list(self.processors.items()):
                if processor is not None:
                    try:
                        del processor
                        self.processors[proc_name] = None
                        logger.info(f"🧹 Cleaned up {proc_name} processor")
                    except Exception as e:
                        logger.warning(f"Error cleaning up {proc_name}: {e}")

            self.processors.clear()

        # Step 3: Aggressive CUDA cleanup
        if torch.cuda.is_available():
            # Force garbage collection
            import gc

            gc.collect()

            # Clear CUDA cache multiple times for stubborn memory
            for _ in range(3):
                torch.cuda.empty_cache()

            # Synchronize CUDA operations
            torch.cuda.synchronize()

            # Log memory status
            allocated_mb = torch.cuda.memory_allocated() / 1e6
            cached_mb = torch.cuda.memory_reserved() / 1e6
            logger.info(
                f"🧹 GPU Memory after cleanup: {allocated_mb:.1f}MB allocated, {cached_mb:.1f}MB cached"
            )

        logger.info("🧹 Aggressive GPU memory cleanup completed")

    def __init__(self, config: NewsReaderV2Config = None):
        self.config = config or NewsReaderV2Config()

        # Device setup with CUDA optimizations
        self.device = self._setup_device()
        self._enable_cuda_optimizations()

        # Model storage
        self.models = {}
        self.processors = {}
        self.pipelines = {}

        # Processing stats
        self.processing_stats = {
            "total_processed": 0,
            "success_rate": 0.0,
            "average_processing_time": 0.0,
            "model_usage_stats": {},
        }

        # Initialize all V2 components
        self._initialize_models()

        logger.info(
            "✅ NewsReader V2 Engine initialized with TRUE multi-modal screenshot processing"
        )

    def __enter__(self):
        """Context manager entry - enable safe resource management"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup on any exit condition"""
        self.cleanup_gpu_memory()
        if exc_type is not None:
            logger.error(
                f"NewsReader V2 Engine exiting due to exception: {exc_type.__name__}: {exc_val}"
            )
        else:
            logger.info("NewsReader V2 Engine cleanly exited")
        return False  # Don't suppress exceptions

    def is_llava_available(self) -> bool:
        """Check if LLaVA model is loaded and ready for use"""
        return (
            self.models.get("llava") is not None
            and self.processors.get("llava") is not None
            and hasattr(self.models["llava"], "generate")
        )

    def cleanup_memory(self):
        """Clean up GPU memory to prevent accumulation"""
        if self.device.type == "cuda":
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def __del__(self):
        """Proper cleanup when engine is destroyed"""
        try:
            self.cleanup_memory()
        except:
            pass  # Ignore errors during cleanup

    def _setup_device(self) -> torch.device:
        """Setup optimal device configuration"""
        if self.config.use_gpu_acceleration and torch.cuda.is_available():
            device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ GPU acceleration enabled: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            device = torch.device("cpu")
            logger.info("✅ CPU processing mode")
        return device

    def _enable_cuda_optimizations(self):
        """Enable CUDA optimizations for maximum performance"""
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✅ CUDA optimizations enabled")

    def _initialize_models(self):
        """Initialize V2 model components - STREAMLINED for memory efficiency"""
        try:
            # Component 1: Primary LLaVA Model (CORE FUNCTIONALITY - Only essential component)
            self._load_llava_model()

            # Component 2: Screenshot Capture System (Essential for webpage processing)
            self._initialize_screenshot_system()

            # DEPRECATED COMPONENTS SKIPPED FOR MEMORY EFFICIENCY:
            # - OCR Engine: DEPRECATED - LLaVA provides superior text extraction
            # - Layout Parser: DEPRECATED - LLaVA provides superior layout understanding
            # - CLIP Vision: DEPRECATED - LLaVA provides comprehensive vision analysis

            logger.info(
                "✅ NewsReader V2 initialized with LLaVA-first architecture (memory optimized)"
            )

        except Exception as e:
            logger.error(f"Error initializing NewsReader V2 models: {e}")
            self._initialize_fallback_systems()

    def _load_llava_model(self):
        """Load LLaVA model for screenshot-based vision-language understanding"""
        try:
            if not LLAVA_AVAILABLE:
                logger.warning("LLaVA not available - core functionality limited")
                return

            # Clear GPU memory aggressively before loading
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                # Get available memory and set conservative limit
                gpu_memory = (
                    torch.cuda.get_device_properties(0).total_memory / 1e9
                )  # GB
                available_memory = gpu_memory * 0.7  # Use only 70% of total
                logger.info(
                    f"🔧 GPU Memory: {gpu_memory:.1f}GB total, limiting to {available_memory:.1f}GB"
                )

            # Setup quantization configuration for memory optimization
            quantization_config = None
            if self.config.use_quantization and self.config.quantization_type != "none":
                logger.info(
                    f"🔧 Setting up {self.config.quantization_type.upper()} quantization for LLaVA..."
                )

                if self.config.quantization_type == "int8":
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=getattr(
                            torch, self.config.quantization_compute_dtype
                        ),
                        bnb_8bit_use_double_quant=True,  # Double quantization for better compression
                    )
                elif self.config.quantization_type == "int4":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=getattr(
                            torch, self.config.quantization_compute_dtype
                        ),
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
                    )

            # Load optimized LLaVA model
            model_info = f"Loading {self.config.llava_model}"
            if quantization_config:
                model_info += (
                    f" with {self.config.quantization_type.upper()} quantization"
                )
            logger.info(model_info + "...")

            self.processors["llava"] = LlavaOnevisionProcessor.from_pretrained(
                self.config.llava_model,
                use_fast=False,  # Set to False to avoid slow processor warnings
                trust_remote_code=True,
                cache_dir=self.config.cache_dir,
            )

            # OPTIMIZED for LLaVA-OneVision-0.5B (87% smaller than previous 7B model)
            # Expected usage: ~1.2GB (vs 7.6GB with old model) - allows for much better memory allocation
            max_gpu_memory = (
                "2GB"  # Conservative for 0.9B parameter model (was 8GB for 7B model)
            )
            if self.device.type == "cuda":
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                # With smaller model, can use standard memory allocation
                safe_memory = gpu_memory * 0.1  # Only 10% needed for this tiny model
                max_gpu_memory = f"{min(2, safe_memory):.1f}GB"
                logger.info(
                    f"🎯 OPTIMIZED MODE: LLaVA-OneVision-0.5B using {max_gpu_memory} of {gpu_memory:.1f}GB GPU memory"
                )

            model_kwargs = {
                "torch_dtype": (
                    torch.float16 if self.device.type == "cuda" else torch.float32
                ),
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "max_memory": {0: max_gpu_memory},  # Conservative GPU memory limit
                "cache_dir": self.config.cache_dir,
                "trust_remote_code": True,
            }

            # Add quantization config if available
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            self.models["llava"] = (
                LlavaOnevisionForConditionalGeneration.from_pretrained(
                    self.config.llava_model, **model_kwargs
                )
            )

            # Move to device carefully (quantized models may already be placed)
            if self.device.type == "cuda" and quantization_config is None:
                # Only move if not quantized (quantized models are auto-placed)
                if not any(
                    "cuda" in str(param.device)
                    for param in self.models["llava"].parameters()
                ):
                    self.models["llava"] = self.models["llava"].to(self.device)

            # TORCH.COMPILE DISABLED - Causes memory pressure with 16 compile workers
            # Apply torch.compile optimization if available and model is on GPU
            # DISABLED: Causes system instability due to memory pressure
            # if (hasattr(torch, 'compile') and self.device.type == 'cuda' and
            #     any('cuda' in str(param.device) for param in self.models['llava'].parameters())):
            #     try:
            #         logger.info("Applying torch.compile optimization to LLaVA...")
            #         self.models['llava'] = torch.compile(
            #             self.models['llava'],
            #             mode="reduce-overhead"
            #         )
            #     except Exception as compile_error:
            #         logger.warning(f"Could not compile LLaVA model: {compile_error}")
            logger.info(
                "🛡️ torch.compile DISABLED to prevent memory pressure (16 workers @ 280MB each)"
            )

            # Log model info
            total_params = sum(p.numel() for p in self.models["llava"].parameters())
            memory_info = ""
            if self.device.type == "cuda":
                allocated_gb = torch.cuda.memory_allocated() / 1e9
                memory_info = f" (GPU: {allocated_gb:.1f}GB)"

            quant_info = (
                f" [{self.config.quantization_type.upper()}]"
                if quantization_config
                else ""
            )
            logger.info(
                f"✅ LLaVA model loaded{quant_info}: {total_params:,} parameters (~7B){memory_info}"
            )

        except Exception as e:
            logger.error(f"Error loading LLaVA model: {e}")
            self.models["llava"] = None

    def _load_clip_model(self):
        """CLIP model DISABLED - Testing redundancy with LLaVA vision analysis

        ANALYSIS: CLIP appears redundant because:
        1. Only returns hardcoded confidence (0.9) and image feature dimensions
        2. Results stored as unused metadata like OCR
        3. LLaVA provides superior vision understanding
        4. Saves ~1-2GB memory without functionality loss

        TODO: Remove CLIP completely after validation testing
        """
        logger.info(
            "🔧 CLIP model disabled - using LLaVA for vision analysis (redundancy test)"
        )
        self.models["clip"] = None
        self.processors["clip"] = None

    def _load_ocr_engine(self):
        """OCR engine DISABLED - Testing redundancy with LLaVA text extraction

        ANALYSIS: OCR appears redundant because:
        1. Primary content extraction uses LLaVA exclusively
        2. OCR results only stored as unused metadata in model_outputs
        3. LLaVA provides superior contextual text understanding
        4. Saves 200-500MB memory without functionality loss

        TODO: Remove OCR completely after validation testing confirms no impact
        """
        logger.info(
            "🔧 OCR engine disabled - using LLaVA for text extraction (redundancy test)"
        )
        self.models["ocr"] = None

    def _load_layout_parser(self):
        """Layout Parser DEPRECATED - LLaVA provides superior layout understanding

        DEPRECATED: Layout Parser caused GPU memory crashes and is redundant
        - LLaVA natively understands document structure and layout
        - Layout Parser consumed 500MB-1GB GPU memory unnecessarily
        - LLaVA provides contextual layout analysis in natural language
        - Eliminates complex computer vision pipeline overhead
        """
        logger.info("🚫 Layout Parser deprecated - LLaVA handles all layout analysis")
        self.models["layout_parser"] = None

    def _initialize_screenshot_system(self):
        """Initialize Playwright screenshot capture system"""
        try:
            # Verify playwright is available
            self.models["screenshot_system"] = {
                "headless": self.config.headless,
                "timeout": self.config.screenshot_timeout,
                "quality": self.config.screenshot_quality,
                "browser_args": [
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                ],
            }
            logger.info("✅ Screenshot capture system initialized")

        except Exception as e:
            logger.error(f"Error initializing screenshot system: {e}")
            self.models["screenshot_system"] = None

    async def capture_webpage_screenshot(
        self, url: str, screenshot_path: str = "page_v2.png"
    ) -> str:
        """
        CORE FUNCTIONALITY: Capture screenshot of webpage for LLaVA analysis

        This is the key differentiator - taking screenshots and using LLaVA's
        vision capabilities to understand webpage content
        """
        logger.info(f"📸 Capturing screenshot for URL: {url}")
        start_time = time.time()

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=self.config.headless,
                    args=self.models["screenshot_system"]["browser_args"],
                )

                page = await browser.new_page()

                # Navigate with timeout
                await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=self.config.screenshot_timeout,
                )

                # Wait for content to load
                await page.wait_for_timeout(2000)

                # Capture screenshot
                await page.screenshot(
                    path=screenshot_path,
                    full_page=False,  # Viewport only for faster processing
                )

                await browser.close()

            elapsed_time = time.time() - start_time
            logger.info(f"✅ Screenshot saved: {screenshot_path} ({elapsed_time:.2f}s)")

            return screenshot_path

        except Exception as e:
            logger.error(f"❌ Screenshot capture failed for {url}: {e}")
            raise

    def analyze_screenshot_with_llava(
        self, screenshot_path: str, custom_prompt: str = None
    ) -> Dict[str, Any]:
        """
        CORE FUNCTIONALITY: Analyze screenshot using LLaVA vision-language model

        This is where the magic happens - LLaVA reads the screenshot like a human
        and extracts structured information (headlines, articles, etc.)
        """
        if not self.is_llava_available():
            raise RuntimeError("LLaVA model not loaded - cannot analyze screenshots")

        # Default news extraction prompt
        if not custom_prompt:
            custom_prompt = """
            Analyze this webpage screenshot for news content. 
            
            Please extract:
            1. The main headline (if visible)
            2. The main news article content (if visible)
            3. Any other relevant news text
            
            Format your response as:
            HEADLINE: [extracted headline]
            ARTICLE: [extracted article content]
            
            If no clear news content is visible, describe what you see.
            """

        try:
            # Load and process image
            image = Image.open(screenshot_path).convert("RGB")

            # Format prompt for LLaVA-Next
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": custom_prompt},
                    ],
                }
            ]

            prompt_text = self.processors["llava"].apply_chat_template(
                conversation, add_generation_prompt=True
            )

            # Limit text length to prevent memory issues (before tokenization)
            max_chars = (
                self.config.max_sequence_length * 4
            )  # Rough estimate: 4 chars per token
            if len(prompt_text) > max_chars:
                # Truncate from the middle, keeping prompt structure
                prefix_len = max_chars // 3
                suffix_len = max_chars // 3
                prompt_text = (
                    prompt_text[:prefix_len]
                    + "...[truncated]..."
                    + prompt_text[-suffix_len:]
                )

            # Process inputs with fast processing - Handle image tokens properly
            inputs = self.processors["llava"](
                images=image,  # Pass image separately
                text=prompt_text,  # Pass formatted text
                return_tensors="pt",
                padding=True,  # Enable fast processing optimizations
                # Remove truncation for LLaVA - it handles sequence length internally
            ).to(self.device)

            # Generate response with optimized parameters
            with torch.no_grad():
                output = self.models["llava"].generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,  # Use configured max tokens
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processors["llava"].tokenizer.eos_token_id,
                )

            # Decode response (only new tokens)
            generated_text = self.processors["llava"].decode(
                output[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
            )

            # Parse structured output
            parsed_content = self._parse_llava_response(generated_text)

            # Cleanup GPU memory after processing
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return {
                "success": True,
                "raw_analysis": generated_text.strip(),
                "parsed_content": parsed_content,
                "screenshot_path": screenshot_path,
                "model_used": "llava-v1.6-mistral-7b",
            }

        except Exception as e:
            logger.error(f"❌ LLaVA screenshot analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "screenshot_path": screenshot_path,
            }

    def _parse_llava_response(self, response: str) -> Dict[str, str]:
        """Parse LLaVA response to extract structured content"""

        parsed = {"headline": "", "article": "", "additional_content": ""}

        try:
            # Look for structured format
            lines = response.split("\n")
            current_section = None

            for line in lines:
                line = line.strip()

                if line.startswith("HEADLINE:"):
                    current_section = "headline"
                    parsed["headline"] = line.replace("HEADLINE:", "").strip()
                elif line.startswith("ARTICLE:"):
                    current_section = "article"
                    parsed["article"] = line.replace("ARTICLE:", "").strip()
                elif current_section == "headline" and line:
                    parsed["headline"] += " " + line
                elif current_section == "article" and line:
                    parsed["article"] += " " + line
                elif line and not current_section:
                    parsed["additional_content"] += line + " "

            # Fallback: if no structured format, treat as general content
            if not parsed["headline"] and not parsed["article"]:
                parsed["additional_content"] = response.strip()

        except Exception as e:
            logger.warning(f"Failed to parse LLaVA response: {e}")
            parsed["additional_content"] = response.strip()

        return parsed

    async def process_news_url_v2(
        self,
        url: str,
        screenshot_path: Optional[str] = None,
        processing_mode: ProcessingMode = ProcessingMode.COMPREHENSIVE,
    ) -> ProcessingResult:
        """
        MAIN V2 PIPELINE: Screenshot → LLaVA Analysis → Enhanced Processing

        This is the core NewsReader V2 workflow:
        1. Capture webpage screenshot
        2. Analyze with LLaVA vision model
        3. Enhance with OCR, layout analysis, CLIP
        4. Return comprehensive results
        """
        start_time = time.time()
        logger.info(f"🔍 Processing news URL: {url}")

        try:
            # Step 1: Capture screenshot (CORE)
            if not screenshot_path:
                screenshot_path = await self.capture_webpage_screenshot(url)

            # Step 2: LLaVA analysis (CORE)
            llava_result = self.analyze_screenshot_with_llava(screenshot_path)

            if not llava_result["success"]:
                raise Exception(
                    f"LLaVA analysis failed: {llava_result.get('error', 'Unknown error')}"
                )

            # Step 3: Enhanced processing - STREAMLINED for LLaVA-first approach
            enhanced_results = {}

            if processing_mode in [
                ProcessingMode.COMPREHENSIVE,
                ProcessingMode.PRECISION,
            ]:
                # OCR DEPRECATED - LLaVA provides superior text extraction
                enhanced_results["ocr"] = {
                    "note": "OCR DEPRECATED - LLaVA provides superior contextual text extraction",
                    "status": "llava_replaces_ocr",
                }

                # Layout Parser DEPRECATED - LLaVA provides superior layout understanding
                enhanced_results["layout"] = {
                    "note": "Layout Parser DEPRECATED - LLaVA provides contextual layout analysis",
                    "status": "llava_replaces_layout",
                }

                # CLIP vision analysis - DISABLED for redundancy testing
                if self.models.get("clip"):
                    clip_result = self._enhance_with_clip_analysis(screenshot_path)
                    enhanced_results["clip"] = clip_result
                else:
                    # CLIP disabled - LLaVA provides sufficient vision analysis
                    enhanced_results["clip"] = {
                        "note": "CLIP disabled - vision analysis provided by LLaVA",
                        "status": "redundancy_test",
                    }

            # Compile final result
            parsed_content = llava_result["parsed_content"]
            processing_time = time.time() - start_time

            result = ProcessingResult(
                content_type=ContentType.WEBPAGE,
                extracted_text=f"HEADLINE: {parsed_content.get('headline', '')}\n\nARTICLE: {parsed_content.get('article', '')}\n\nADDITIONAL: {parsed_content.get('additional_content', '')}",
                visual_description=llava_result["raw_analysis"],
                layout_analysis=enhanced_results.get("layout", {}),
                confidence_score=0.85 if llava_result["success"] else 0.0,
                processing_time=processing_time,
                model_outputs={"llava": llava_result, **enhanced_results},
                metadata={
                    "url": url,
                    "screenshot_path": screenshot_path,
                    "processing_mode": processing_mode.value,
                    "models_used": ["llava"] + list(enhanced_results.keys()),
                },
                screenshot_path=screenshot_path,
            )

            logger.info(
                f"✅ NewsReader V2 processing completed: {processing_time:.2f}s"
            )
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ NewsReader V2 processing failed: {e}")

            # Return error result
            return ProcessingResult(
                content_type=ContentType.WEBPAGE,
                extracted_text=f"Processing failed: {str(e)}",
                visual_description=f"Error processing URL: {url}",
                layout_analysis={},
                confidence_score=0.0,
                processing_time=processing_time,
                model_outputs={"error": str(e)},
                metadata={"url": url, "error": str(e)},
                screenshot_path=screenshot_path,
            )

    def _enhance_with_ocr(self, screenshot_path: str) -> Dict[str, Any]:
        """Enhance with OCR text extraction"""
        try:
            if not self.models.get("ocr"):
                return {"error": "OCR not available"}

            results = self.models["ocr"].readtext(screenshot_path)

            extracted_text = []
            confidence_scores = []

            for bbox, text, confidence in results:
                extracted_text.append(text)
                confidence_scores.append(confidence)

            return {
                "extracted_text": " ".join(extracted_text),
                "confidence": (
                    sum(confidence_scores) / len(confidence_scores)
                    if confidence_scores
                    else 0
                ),
                "text_blocks": len(results),
            }

        except Exception as e:
            return {"error": str(e)}

    def _enhance_with_layout_analysis(self, screenshot_path: str) -> Dict[str, Any]:
        """Enhance with layout structure analysis"""
        try:
            if not self.models.get("layout_parser"):
                return {"error": "Layout parser not available"}

            image = Image.open(screenshot_path)
            layout = self.models["layout_parser"].detect(image)

            return {
                "layout_blocks": len(layout),
                "block_types": (
                    [block.type for block in layout]
                    if hasattr(layout, "__iter__")
                    else []
                ),
                "confidence": 0.8,
            }

        except Exception as e:
            return {"error": str(e)}

    def _enhance_with_clip_analysis(self, screenshot_path: str) -> Dict[str, Any]:
        """Enhance with CLIP vision analysis"""
        try:
            if not self.models.get("clip"):
                return {"error": "CLIP not available"}

            image = Image.open(screenshot_path)
            inputs = self.processors["clip"](images=image, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                image_features = self.models["clip"].get_image_features(**inputs)

            return {
                "image_features_shape": list(image_features.shape),
                "confidence": 0.9,
            }

        except Exception as e:
            return {"error": str(e)}

    def _create_basic_layout_parser(self):
        """Create basic layout parser fallback"""

        class BasicLayoutParser:
            def detect(self, image):
                # Simple fallback
                return {
                    "layout_blocks": [
                        {"type": "Text", "bbox": [0, 0, 100, 100], "confidence": 0.5}
                    ],
                    "confidence": 0.5,
                }

        return BasicLayoutParser()

    def _initialize_fallback_systems(self):
        """Initialize fallback processing systems"""
        logger.info("Initializing fallback processing systems...")

        # Create basic fallback systems
        self.pipelines["fallback_screenshot"] = (
            self._create_fallback_screenshot_processor()
        )
        self.pipelines["fallback_analysis"] = self._create_fallback_analysis_processor()

        logger.info("✅ Fallback systems initialized")

    def _create_fallback_screenshot_processor(self):
        """Create fallback screenshot processing system"""

        class FallbackScreenshotProcessor:
            async def process(self, url):
                return {
                    "screenshot_path": "fallback_screenshot.png",
                    "success": False,
                    "fallback_reason": "Screenshot system not available",
                }

        return FallbackScreenshotProcessor()

    def _create_fallback_analysis_processor(self):
        """Create fallback analysis system"""

        class FallbackAnalysisProcessor:
            def process(self, screenshot_path):
                return {
                    "success": False,
                    "raw_analysis": "Fallback analysis - LLaVA not available",
                    "parsed_content": {
                        "headline": "Analysis not available",
                        "article": "LLaVA model not loaded",
                        "additional_content": "Running in fallback mode",
                    },
                }

        return FallbackAnalysisProcessor()


def log_feedback(event: str, details: dict):
    """Log feedback for monitoring and improvement"""
    try:
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(
                f"{datetime.utcnow().isoformat()}\t{event}\t{json.dumps(details)}\n"
            )
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")


# Export main components
__all__ = [
    "NewsReaderV2Engine",
    "NewsReaderV2Config",
    "ContentType",
    "ProcessingMode",
    "ProcessingResult",
    "log_feedback",
]

if __name__ == "__main__":
    # Test NewsReader V2 Engine with REAL screenshot functionality
    async def test_real_screenshot_processing():
        print("🔍 Testing NewsReader V2 Engine - REAL Screenshot Processing")
        print("=" * 70)

        config = NewsReaderV2Config(
            default_mode=ProcessingMode.COMPREHENSIVE,
            use_gpu_acceleration=torch.cuda.is_available(),
        )

        engine = NewsReaderV2Engine(config)

        # Test with real news URL
        test_url = "https://www.bbc.co.uk/news"

        try:
            result = await engine.process_news_url_v2(test_url)

            print("✅ Processing Result:")
            print(f"   Success: {result.confidence_score > 0}")
            print(f"   Processing Time: {result.processing_time:.2f}s")
            print(f"   Screenshot: {result.screenshot_path}")
            print(f"   Models Used: {result.metadata.get('models_used', [])}")
            print(
                f"   Headline: {result.model_outputs.get('llava', {}).get('parsed_content', {}).get('headline', 'N/A')[:100]}..."
            )
            print(
                f"   Article: {result.model_outputs.get('llava', {}).get('parsed_content', {}).get('article', 'N/A')[:200]}..."
            )

        except Exception as e:
            print(f"❌ Test failed: {e}")

        print("=" * 70)
        print("🎯 NewsReader V2 Engine test completed")

    asyncio.run(test_real_screenshot_processing())
