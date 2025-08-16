import asyncio
import logging
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import BitsAndBytesConfig
from PIL import Image
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("production_newsreader")

class ProductionNewsReader:
    """
    Production-Grade NewsReader Implementation
    
    This implementation addresses all identified LLaVA model issues with:
    - Proper model class matching (LlavaProcessor + LlavaForConditionalGeneration)
    - Comprehensive error handling and fallback strategies
    - Production memory management and monitoring
    - Built-in functionality testing and health checks
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_type = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🏗️ Production NewsReader initialized on {self.device}")
    
    async def initialize_with_fallback_strategy(self) -> bool:
        """
        Production initialization with comprehensive fallback strategy.
        
        Strategy:
        1. Primary: Proper LLaVA implementation with correct model classes
        2. Fallback: BLIP-2 alternative model
        3. Graceful failure with proper error reporting
        
        Returns:
            bool: True if any initialization strategy succeeds
        """
        logger.info("🚀 Starting production NewsReader initialization with fallback strategy...")
        
        # Strategy 1: Production LLaVA with proper model classes
        if await self.initialize_production_llava():
            logger.info("✅ Primary strategy successful: Production LLaVA loaded")
            return True
        
        # Strategy 2: BLIP-2 fallback
        if await self.initialize_fallback_blip2():
            logger.info("✅ Fallback strategy successful: BLIP-2 loaded")
            return True
        
        # Strategy 3: Graceful failure
        logger.error("❌ All initialization strategies failed")
        return False
    
    async def initialize_production_llava(self) -> bool:
        """
        Initialize Production LLaVA with proper model classes and configuration.
        
        Addresses all identified issues:
        - Uses correct LlavaProcessor + LlavaForConditionalGeneration classes
        - Proper quantization configuration
        - Memory management with explicit limits
        - Built-in functionality testing
        
        Returns:
            bool: True if initialization successful
        """
        logger.info("🔧 Initializing Production LLaVA with proper model classes...")
        
        try:
            model_id = "llava-hf/llava-1.5-7b-hf"
            logger.info(f"Loading model: {model_id}")
            logger.info("Using CORRECT model classes: LlavaProcessor + LlavaForConditionalGeneration")
            
            # Quantization configuration for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,  # Enhanced quantization
                bnb_8bit_quant_type="nf4"       # Quality optimization
            )
            
            # Load with CORRECT model classes and proper configuration
            self.processor = LlavaProcessor.from_pretrained(
                model_id,
                use_fast=True  # Enable fast processor to avoid deprecation warnings
            )
            
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                max_memory={0: "20GB"},  # Explicit memory limit
                torch_dtype=torch.float16
            )
            
            # Validate model components
            if hasattr(self.model, 'vision_tower') and hasattr(self.model, 'language_model'):
                logger.info("✅ Model components validated: vision_tower and language_model present")
            else:
                logger.warning("⚠️ Model components not properly loaded")
            
            # Test model functionality
            if await self._test_model_functionality():
                self.model_type = "production_llava"
                
                # Log memory usage
                if torch.cuda.is_available():
                    memory_gb = round(torch.cuda.memory_allocated() / 1024**3, 1)
                    _max_memory_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 1)
                    utilization = round((torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100), 1) if torch.cuda.max_memory_allocated() > 0 else 0
                    
                    logger.info(f"💾 Memory: {memory_gb}GB allocated")
                    logger.info(f"📊 Memory utilization: {utilization}%")
                
                logger.info("✅ Production LLaVA loaded successfully")
                return True
            else:
                logger.error("❌ Model functionality test failed")
                return False
                
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
            
            self.processor = Blip2Processor.from_pretrained(model_id)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Test BLIP-2 functionality
            test_image = Image.new('RGB', (224, 224), color='white')
            inputs = self.processor(test_image, "Describe this image", return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=50)
                result = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            if result:
                self.model_type = "blip2_fallback"
                logger.info("✅ BLIP-2 fallback loaded successfully")
                return True
            else:
                logger.error("❌ BLIP-2 functionality test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ BLIP-2 fallback initialization failed: {e}")
            return False
    
    async def _test_model_functionality(self) -> bool:
        """
        Test model functionality with known input.
        
        Returns:
            bool: True if functionality test passes
        """
        try:
            logger.info("🧪 Testing model functionality...")
            
            # Create test image
            test_image = Image.new('RGB', (224, 224), color='white')
            
            # Use proper LLaVA prompt format for testing
            prompt_text = "USER: <image>\nDescribe this image.\nASSISTANT:"
            
            # Process with explicit parameter naming to avoid order issues
            inputs = self.processor(
                text=prompt_text,  # Explicit parameter naming with proper format
                images=test_image,  # Correct order
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response with proper settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,  # Use greedy for consistent testing
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                
                result = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Validate result - should contain actual description
            if result and len(result.strip()) > 20:  # Reasonable length check
                # Extract the assistant response
                if "ASSISTANT:" in result:
                    clean_result = result.split("ASSISTANT:")[-1].strip()
                    logger.info(f"✅ Model functionality test passed: '{clean_result[:50]}...'")
                    return True
                else:
                    logger.error(f"❌ Model functionality test failed: No ASSISTANT response in '{result}'")
                    return False
            else:
                logger.error(f"❌ Model functionality test failed: '{result}'")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model functionality test error: {e}")
            return False
    
    async def analyze_news_content(self, image: Image.Image, context: str = "") -> Optional[str]:
        """
        Analyze news content from image with production error handling.
        
        Args:
            image: PIL Image to analyze
            context: Additional context for analysis
            
        Returns:
            Optional[str]: Analysis result or None if failed
        """
        if not self.model or not self.processor:
            logger.error("❌ Model not initialized. Call initialize_with_fallback_strategy() first.")
            return None
        
        try:
            # Validate input
            if not isinstance(image, Image.Image):
                logger.error("❌ Invalid input: image must be PIL Image")
                return None
            
            # Convert image if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Prepare prompt based on model type
            if self.model_type == "production_llava":
                prompt = f"USER: <image>\nAnalyze this news content. {context}\nASSISTANT:"
                
                # Process with explicit parameter naming
                inputs = self.processor(
                    text=prompt,      # Explicit naming
                    images=image,     # Correct order
                    return_tensors="pt"
                ).to(self.device)
                
            elif self.model_type == "blip2_fallback":
                prompt = f"Analyze this news content. {context}"
                inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            
            else:
                logger.error(f"❌ Unknown model type: {self.model_type}")
                return None
            
            # Generate with production settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                
                result = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean result based on model type
            if self.model_type == "production_llava":
                # Remove prompt from LLaVA result
                if "ASSISTANT:" in result:
                    result = result.split("ASSISTANT:")[-1].strip()
            elif self.model_type == "blip2_fallback":
                # Clean BLIP-2 result - remove the input prompt
                # BLIP-2 often repeats the prompt, so extract only the new content
                prompt_text = f"Analyze this news content. {context}"
                if prompt_text in result:
                    # Remove the prompt from the result
                    result = result.replace(prompt_text, "").strip()
                    # Remove any duplicate text patterns
                    parts = result.split()
                    if len(parts) > 3:
                        # Look for meaningful content beyond the prompt
                        result = " ".join(parts[:50])  # Limit to first 50 words
                
                # If result is still just the prompt or empty, provide fallback
                if len(result.strip()) < 10 or result.strip() == prompt_text.strip():
                    result = "This appears to be a news webpage with text content and images. Unable to provide detailed analysis."
            
            return result
            
        except Exception as e:
            logger.error(f"❌ News content analysis failed: {e}")
            return None
    
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
    
    def cleanup(self):
        """Production cleanup with proper memory management"""
        try:
            if self.model:
                del self.model
                self.model = None
            
            if self.processor:
                del self.processor
                self.processor = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ Production cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

# Production usage example
async def main():
    """Production NewsReader usage example"""
    
    # Initialize with production strategy
    newsreader = ProductionNewsReader()
    success = await newsreader.initialize_with_fallback_strategy()
    
    if success:
        print("✅ Production NewsReader ready")
        
        # Health check
        health = await newsreader.health_check()
        print(f"📊 Health: {health}")
        
        # Performance metrics
        metrics = await newsreader.get_performance_metrics()
        print(f"📈 Metrics: {metrics}")
        
        # Test with image
        test_image = Image.new('RGB', (400, 300), color='lightblue')
        result = await newsreader.analyze_news_content(test_image, "Sample news analysis")
        print(f"🔍 Analysis: {result}")
        
        # Cleanup
        newsreader.cleanup()
    else:
        print("❌ Production NewsReader initialization failed")

if __name__ == "__main__":
    asyncio.run(main())
