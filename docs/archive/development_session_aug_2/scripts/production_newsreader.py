#!/usr/bin/env python3
"""
Production-Grade NewsReader with Proper LLaVA Implementation

This file was moved from archive_obsolete_files to docs/archive to prevent
linting and parse errors during CI. Original content preserved below.
"""

# --- Begin original file content ---
#!/usr/bin/env python3
"""
Production-Grade NewsReader with Proper LLaVA Implementation

This implements a robust, production-ready solution addressing all identified issues:
1. Correct         except Exception as e:
            logger.error(f"❌ Production LLaVA initialization failed: {e}")
            return False
    
    async def health_check(self) -> dict:
        """Production health check endpoint"""
        try:
            health_status = {
                "status": "healthy" if self.model else "unhealthy",
                "model_type": self.model_type,
                "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 1) if torch.cuda.is_available() else 0,
                "memory_utilization_percent": round((torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100), 1) if torch.cuda.is_available() and torch.cuda.max_memory_allocated() > 0 else 0,
                "gpu_available": torch.cuda.is_available()
            }
            
            # Test basic functionality if model is loaded
            if self.model and self.processor:
                test_image = Image.new('RGB', (224, 224), color='white')
                test_result = await self.analyze_news_content(test_image, "Test health check")
                health_status["functionality_test"] = "passed" if test_result else "failed"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_performance_metrics(self) -> dict:
        """Get production performance metrics"""
        return {
            "model_type": self.model_type,
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 1) if torch.cuda.is_available() else 0,
            "gpu_utilization": torch.cuda.utilization() if torch.cuda.is_available() else 0,
            "model_loaded": bool(self.model),
            "processor_loaded": bool(self.processor)
        }
    
    async def _test_model_functionality(self) -> bool: class matching for LLaVA-1.5
2. Proper error handling and fallback strategies  
3. Comprehensive compatibility testing
4. Memory management optimization
5. Future-proof API usage

No quick fixes - only production-grade solutions.
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
import torch
from transformers import (
    LlavaProcessor, 
    LlavaForConditionalGeneration,
    BlipProcessor,
    BlipForConditionalGeneration,
    BitsAndBytesConfig
)
from PIL import Image
import requests
from fastapi import FastAPI
from pydantic import BaseModel

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ToolCall(BaseModel):
    args: list
    kwargs: dict

class ProductionNewsReader:
    """
    Production-grade NewsReader with robust LLaVA implementation.
    
    Key Features:
    - Proper model class matching (LlavaProcessor + LlavaForConditionalGeneration)
    - Comprehensive error handling and recovery
    - Multiple fallback strategies
    - Memory optimization and monitoring
    - Future-proof API compatibility
    - Production logging and monitoring
    """
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = None
        self.initialization_successful = False
        self.performance_metrics = {}
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get comprehensive GPU memory usage metrics."""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "memory_utilization_percent": (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
            }
        return {"cpu_only": True}
    
    async def initialize_production_llava(self) -> bool:
        """
        Initialize LLaVA with proper model classes and comprehensive error handling.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        logger.info("🔧 Initializing Production LLaVA with proper model classes...")
        
        try:
            # Production-grade quantization configuration
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,  # Additional quantization for memory efficiency
                bnb_8bit_quant_type="nf4"       # Normalized float 4-bit for better quality
            )
            
            # Use correct model ID and classes for LLaVA-1.5
            model_id = "llava-hf/llava-1.5-7b-hf"
            
            logger.info(f"Loading model: {model_id}")
            logger.info("Using CORRECT model classes: LlavaProcessor + LlavaForConditionalGeneration")
            
            # Initialize processor with proper settings
            self.processor = LlavaProcessor.from_pretrained(
                model_id,
                use_fast=True,  # Enable fast processor to avoid deprecation warning
                trust_remote_code=True
            )
            
            # Initialize model with proper class matching
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "20GB"}  # Explicit memory limit for safety
            )
            
            # Validate model initialization
            if hasattr(self.model, 'vision_tower') and hasattr(self.model, 'language_model'):
                logger.info("✅ Model components validated: vision_tower and language_model present")
            else:
                logger.warning("⚠️ Model components validation failed")
                return False
            
            # Test basic functionality
            test_successful = await self._test_model_functionality()
            if not test_successful:
                logger.error("❌ Model functionality test failed")
                return False
            
            self.model_type = "llava-1.5-production"
            self.initialization_successful = True
            
            # Log memory usage
            memory_usage = self.get_memory_usage()
            logger.info(f"✅ Production LLaVA loaded successfully")
            logger.info(f"💾 Memory: {memory_usage.get('allocated_gb', 0):.1f}GB allocated")
            logger.info(f"📊 Memory utilization: {memory_usage.get('memory_utilization_percent', 0):.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Production LLaVA initialization failed: {e}")
            return False
    
    async def initialize_fallback_blip2(self) -> bool:
        """
        Initialize BLIP-2 as production fallback model.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        logger.info("🔧 Initializing BLIP-2 fallback model...")
        
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            
            model_id = "Salesforce/blip2-opt-2.7b"
            
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Test basic functionality
            test_successful = await self._test_model_functionality()
            if not test_successful:
                logger.error("❌ BLIP-2 functionality test failed")
                return False
            
            self.model_type = "blip2-fallback"
            self.initialization_successful = True
            
            memory_usage = self.get_memory_usage()
            logger.info(f"✅ BLIP-2 fallback loaded successfully")
            logger.info(f"💾 Memory: {memory_usage.get('allocated_gb', 0):.1f}GB allocated")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ BLIP-2 fallback initialization failed: {e}")
            return False
    
    async def _test_model_functionality(self) -> bool:
        """
        Test basic model functionality with a simple image.
        
        Returns:
            bool: True if test passes, False otherwise
        """
        try:
            logger.info("🧪 Testing model functionality...")
            
            # Create a simple test image
            test_image = Image.new('RGB', (224, 224), color='white')
            test_prompt = "Describe this image briefly."
            
            # Test processing based on model type
            if hasattr(self.processor, 'apply_chat_template'):
                # LLaVA format
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": test_prompt}
                        ]
                    }
                ]
                prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(prompt_text, test_image, return_tensors="pt").to(self.device)
            else:
                # BLIP-2 format
                inputs = self.processor(test_image, test_prompt, return_tensors="pt").to(self.device)
            
            # Test generation
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,  # Deterministic for testing
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Test decoding
            if hasattr(self.processor, 'apply_chat_template'):
                generated_text = self.processor.decode(
                    output[0][len(inputs.input_ids[0]):], 
                    skip_special_tokens=True
                )
            else:
                generated_text = self.processor.decode(output[0], skip_special_tokens=True)
            
            if len(generated_text.strip()) > 0:
                logger.info(f"✅ Model functionality test passed: '{generated_text[:50]}...'")
                return True
            else:
                logger.error("❌ Model functionality test failed: empty output")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model functionality test failed: {e}")
            return False
    
    async def initialize_with_fallback_strategy(self) -> bool:
        """
        Production initialization with comprehensive fallback strategy.
        
        Returns:
            bool: True if any model successfully initialized
        """
        logger.info("🚀 Starting production NewsReader initialization with fallback strategy...")
        
        # Strategy 1: Try production LLaVA with proper classes
        success = await self.initialize_production_llava()
        if success:
            logger.info("✅ Primary strategy successful: Production LLaVA loaded")
            return True
        
        logger.warning("⚠️ Primary strategy failed, trying fallback...")
        
        # Strategy 2: Fallback to BLIP-2
        success = await self.initialize_fallback_blip2()
        if success:
            logger.info("✅ Fallback strategy successful: BLIP-2 loaded")
            return True
        
        logger.error("❌ All initialization strategies failed")
        return False
    
    async def analyze_image_production(self, image_input, prompt: str = None) -> Dict[str, Any]:
        """
        Production-grade image analysis with comprehensive error handling.
        
        Args:
            image_input: PIL Image, file path, or URL
            prompt: Analysis prompt (uses default if None)
        
        Returns:
            Dict containing analysis results and metadata
        """
        if not self.initialization_successful:
            return {
                "success": False,
                "error": "Model not properly initialized",
                "model_type": None
            }
        
        # Default news analysis prompt
        if not prompt:
            prompt = (
                "Analyze this webpage screenshot. Extract the main headline and summarize "
                "the primary news content in 2-3 sentences. Identify if this appears to be "
                "a legitimate news article."
            )
        
        try:
            # Handle different image input types
            if isinstance(image_input, str):
                if image_input.startswith(('http://', 'https://')):
                    # URL input
                    response = requests.get(image_input, timeout=10)
                    response.raise_for_status()
                    image = Image.open(response.content).convert("RGB")
                else:
                    # File path input
                    image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                # PIL Image input
                image = image_input.convert("RGB")
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Process based on model type
            start_time = asyncio.get_event_loop().time()
            
            if self.model_type == "llava-1.5-production":
                result = await self._analyze_with_llava(image, prompt)
            elif self.model_type == "blip2-fallback":
                result = await self._analyze_with_blip2(image, prompt)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Add metadata
            result.update({
                "model_type": self.model_type,
                "processing_time_seconds": processing_time,
                "memory_usage": self.get_memory_usage()
            })
            
            logger.info(f"✅ Image analysis completed in {processing_time:.2f}s using {self.model_type}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Image analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_type": self.model_type
            }
    
    async def _analyze_with_llava(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Analyze image using LLaVA with proper API usage."""
        
        # Use proper LLaVA conversation format
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template properly
        prompt_text = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True
        )
        
        # Process inputs with correct parameter order
        inputs = self.processor(
            text=prompt_text,  # Explicit parameter naming
            images=image,      # Correct order: images before text in processor
            return_tensors="pt"
        ).to(self.device)
        
        # Generate with production settings
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode properly (only new tokens for LLaVA)
        generated_text = self.processor.decode(
            output[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()
        
        return {
            "success": True,
            "analysis": generated_text,
            "prompt_used": prompt
        }
    
    async def _analyze_with_blip2(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Analyze image using BLIP-2."""
        
        # Process inputs
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7
            )
        
        # Decode (full output for BLIP-2)
        generated_text = self.processor.decode(output[0], skip_special_tokens=True).strip()
        
        return {
            "success": True,
            "analysis": generated_text,
            "prompt_used": prompt
        }

# Global instance for production use
production_newsreader = ProductionNewsReader()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Production FastAPI lifespan with robust initialization."""
    logger.info("🚀 Starting Production NewsReader Service...")
    
    success = await production_newsreader.initialize_with_fallback_strategy()
    if not success:
        logger.error("❌ Failed to initialize any model - service unavailable")
        raise RuntimeError("NewsReader initialization failed")
    
    logger.info("✅ Production NewsReader Service ready")
    yield
    
    logger.info("🛑 Shutting down Production NewsReader Service...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Production FastAPI app
app = FastAPI(
    title="Production NewsReader Service",
    description="Robust, production-grade NewsReader with proper LLaVA implementation",
    version="2.0.0",
    lifespan=lifespan
)

@app.post("/analyze_image")
async def analyze_image_endpoint(call: ToolCall):
    """Production image analysis endpoint."""
    return await production_newsreader.analyze_image_production(*call.args, **call.kwargs)

@app.get("/health")
async def health():
    """Comprehensive health check."""
    memory_usage = production_newsreader.get_memory_usage()
    return {
        "status": "healthy" if production_newsreader.initialization_successful else "unhealthy",
        "model_type": production_newsreader.model_type,
        "memory_usage": memory_usage,
        "initialization_successful": production_newsreader.initialization_successful
    }

@app.get("/system_status")
async def system_status():
    """Detailed system status for monitoring."""
    return {
        "model_loaded": production_newsreader.model is not None,
        "model_type": production_newsreader.model_type,
        "initialization_successful": production_newsreader.initialization_successful,
        "memory_usage": production_newsreader.get_memory_usage(),
        "cuda_available": torch.cuda.is_available(),
        "device": production_newsreader.device
    }

async def test_production_implementation():
    """Comprehensive production testing."""
    print("🧪 Testing Production NewsReader Implementation")
    print("=" * 60)
    
    # Test initialization
    print("\n📊 Testing robust initialization...")
    success = await production_newsreader.initialize_with_fallback_strategy()
    
    if success:
        print(f"✅ Initialization successful with {production_newsreader.model_type}")
        
        # Test functionality
        print("\n🖼️ Testing image analysis...")
        test_image = Image.new('RGB', (800, 600), color='lightblue')
        
        result = await production_newsreader.analyze_image_production(
            test_image,
            "Analyze this test image and describe what you see."
        )
        
        if result.get('success'):
            print(f"✅ Analysis successful: {result.get('analysis', '')[:100]}...")
            print(f"⏱️ Processing time: {result.get('processing_time_seconds', 0):.2f}s")
            print(f"💾 Memory usage: {result.get('memory_usage', {}).get('allocated_gb', 0):.1f}GB")
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
    else:
        print("❌ Initialization failed - no models available")
    
    print("\n" + "=" * 60)
    print("🎯 Production testing complete")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_production_implementation())
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8005)

# --- End original file content ---
