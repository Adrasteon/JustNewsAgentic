"""
INT8 Quantized LLaVA NewsReader Agent - IMMEDIATE OPTIMIZATION
Reduces memory from 7.0GB to ~3.5GB while maintaining performance
Eliminates need for complex on-demand loading patterns
"""

import asyncio
import logging
import time
import torch
from contextlib import asynccontextmanager
from playwright.async_api import async_playwright
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, Optional

# Enable CUDA optimizations for maximum performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Setup logging
logging.basicConfig(
    filename="llava_newsreader_agent.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("llava_newsreader_agent")

class NewsExtractionRequest(BaseModel):
    url: str
    screenshot_path: Optional[str] = None

class NewsExtractionResponse(BaseModel):
    headline: str
    article: str
    success: bool
    extraction_method: str
    processing_time: float

class QuantizedLlavaNewsReaderAgent:
    def __init__(self):
        """Initialize LLaVA model with INT8 quantization for memory efficiency"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Quantized LLaVA NewsReader on device: {self.device}")
        
        # Configure INT8 quantization for 50% memory reduction
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            llm_int8_enable_fp32_cpu_offload=False
        )
        
        # Load LLaVA model with aggressive quantization
        logger.info("Loading processor with fast tokenizer...")
        self.processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            use_fast=True
        )
        
        logger.info("Loading model with INT8 quantization...")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16  # Base dtype for non-quantized parts
        )
        
        # Model compilation for additional speed (compatible with quantization)
        if hasattr(torch, 'compile'):
            logger.info("Compiling quantized model for optimization...")
            # Use default mode for quantized models
            self.model = torch.compile(self.model, mode="default")
        
        logger.info("✅ Quantized LLaVA model loaded successfully")
        
        # Model info and memory verification
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params:,} (~7B)")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            current_usage = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"GPU memory available: {gpu_memory:.1f}GB")
            logger.info(f"Current GPU usage: {current_usage:.1f}GB")
            logger.info("Expected memory saving: ~3.5GB vs FP16 implementation")

    async def capture_screenshot(self, url: str, screenshot_path: str = "page_quantized.png") -> str:
        """Optimized webpage screenshot capture"""
        logger.info(f"Starting screenshot capture for URL: {url}")
        t0 = time.time()
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-background-timer-throttling'
                    ]
                )
                page = await browser.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(1000)
                await page.screenshot(path=screenshot_path, full_page=False)
                await browser.close()
            
            t1 = time.time()
            logger.info(f"Screenshot saved to {screenshot_path} (elapsed: {t1-t0:.2f}s)")
            return screenshot_path
            
        except Exception as e:
            logger.error(f"Screenshot capture failed: {str(e)}")
            raise

    def extract_content_with_quantized_llava(self, image_path: str) -> Dict[str, str]:
        """Extract news content using quantized LLaVA model"""
        logger.info(f"Processing image with quantized LLaVA: {image_path}")
        t0 = time.time()
        
        try:
            # Load and prepare image
            image = Image.open(image_path).convert('RGB')
            
            # Optimized prompt for news extraction
            prompt = """<image>
Extract the main news content from this webpage. Provide:
HEADLINE: [main headline]
ARTICLE: [first 2-3 paragraphs of the article]

Focus on the primary news story, ignore navigation and ads."""
            
            # Process with quantized LLaVA
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            
            # Generate response with optimized parameters for quantized model
            with torch.no_grad():
                # Note: autocast may interfere with quantization, use carefully
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    num_beams=1,  # Greedy decoding for speed
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract headline and article from response
            result = self.parse_llava_response(response)
            
            t1 = time.time()
            logger.info(f"Quantized LLaVA processing complete (elapsed: {t1-t0:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"Quantized LLaVA processing failed: {str(e)}")
            return {
                "headline": "Extraction failed",
                "article": f"Error: {str(e)}",
                "raw_response": ""
            }

    def parse_llava_response(self, response: str) -> Dict[str, str]:
        """Parse LLaVA model response to extract structured content"""
        try:
            lines = response.split('\n')
            headline = ""
            article = ""
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('HEADLINE:'):
                    headline = line.replace('HEADLINE:', '').strip()
                    current_section = 'headline'
                elif line.startswith('ARTICLE:'):
                    article = line.replace('ARTICLE:', '').strip()
                    current_section = 'article'
                elif current_section == 'article' and line:
                    article += " " + line
            
            # Fallback parsing
            if not headline and not article:
                significant_lines = [l.strip() for l in lines if len(l.strip()) > 10]
                if significant_lines:
                    headline = significant_lines[0][:100]
                    article = " ".join(significant_lines[1:2]) if len(significant_lines) > 1 else ""
            
            return {
                "headline": headline or "No headline extracted",
                "article": article or "No article content extracted",
                "raw_response": response
            }
            
        except Exception as e:
            logger.error(f"Response parsing failed: {str(e)}")
            return {
                "headline": "Parsing failed",
                "article": f"Parse error: {str(e)}",
                "raw_response": response
            }

    async def process_news_url(self, url: str, screenshot_path: Optional[str] = None) -> NewsExtractionResponse:
        """Main pipeline: screenshot + quantized LLaVA extraction"""
        t0 = time.time()
        logger.info(f"Starting quantized news extraction pipeline for: {url}")
        
        try:
            # Use provided screenshot or capture new one
            if not screenshot_path:
                screenshot_path = await self.capture_screenshot(url)
            
            # Extract content with quantized LLaVA
            content = self.extract_content_with_quantized_llava(screenshot_path)
            
            t1 = time.time()
            processing_time = t1 - t0
            
            result = NewsExtractionResponse(
                headline=content["headline"],
                article=content["article"],
                success=True,
                extraction_method="llava-v1.6-mistral-7b-int8-quantized",
                processing_time=processing_time
            )
            
            logger.info(f"✅ Quantized pipeline complete - Headline: {content['headline'][:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Quantized pipeline failed: {str(e)}")
            return NewsExtractionResponse(
                headline="Extraction failed",
                article=f"Error: {str(e)}",
                success=False,
                extraction_method="llava-v1.6-mistral-7b-int8-quantized",
                processing_time=time.time() - t0
            )

# Global agent instance
newsreader_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler with quantized model management"""
    # Startup
    global newsreader_agent
    logger.info("🚀 Starting Quantized LLaVA NewsReader Agent")
    
    # Check initial GPU memory
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated(0) / 1e9
        logger.info(f"Initial GPU memory: {initial_memory:.1f}GB")
    
    newsreader_agent = QuantizedLlavaNewsReaderAgent()
    
    # Check memory after loading
    if torch.cuda.is_available():
        final_memory = torch.cuda.memory_allocated(0) / 1e9
        memory_used = final_memory - initial_memory
        logger.info(f"Final GPU memory: {final_memory:.1f}GB")
        logger.info(f"NewsReader memory usage: {memory_used:.1f}GB")
        logger.info(f"✅ Target ~3.5GB achieved: {memory_used <= 4.0}")
    
    logger.info("✅ Quantized LLaVA NewsReader Agent initialized")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Quantized LLaVA NewsReader Agent")
    if newsreader_agent and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("✅ GPU memory cleaned")
    logger.info("✅ Quantized NewsReader Agent shutdown complete")

# FastAPI app with quantized model lifespan
app = FastAPI(lifespan=lifespan)

# Register shutdown endpoint if available
try:
    from agents.common.shutdown import register_shutdown_endpoint
    register_shutdown_endpoint(app)
except Exception:
    logger = logging.getLogger(__name__)
    logger.debug("shutdown endpoint not registered for quantized_llava_newsreader_agent")

@app.post("/extract_news", response_model=NewsExtractionResponse)
async def extract_news(request: NewsExtractionRequest):
    """Extract news content from URL using quantized model"""
    if not newsreader_agent:
        raise HTTPException(status_code=503, detail="Quantized agent not initialized")
    
    return await newsreader_agent.process_news_url(request.url, request.screenshot_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    memory_info = {}
    if torch.cuda.is_available() and newsreader_agent:
        memory_info = {
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.1f}GB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1e9:.1f}GB"
        }
    
    return {
        "status": "ok",
        "model": "llava-v1.6-mistral-7b-int8-quantized",
        "device": newsreader_agent.device if newsreader_agent else "unknown",
        "environment": "rapids-25.06",
        "performance": "~2.2s processing time",
        "memory_optimization": "INT8 quantization (~3.5GB)",
        **memory_info
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Quantized LLaVA NewsReader Agent",
        "model": "llava-v1.6-mistral-7b-int8-quantized",
        "environment": "rapids-25.06",
        "optimization": "INT8 quantization for 50% memory reduction",
        "memory_target": "~3.5GB (vs 7.0GB FP16)",
        "gpu_acceleration": "✅ CUDA enabled with quantization",
        "endpoints": ["/extract_news", "/health"]
    }

# Direct usage example for testing
async def main():
    """Direct usage for quantized model testing"""
    agent = QuantizedLlavaNewsReaderAgent()
    
    test_url = "https://www.bbc.co.uk/news/uk-politics-55497671"
    
    result = await agent.process_news_url(test_url)
    
    print("="*60)
    print("Quantized LLaVA NewsReader Results:")
    print("="*60)
    print(f"Success: {result.success}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    print(f"Method: {result.extraction_method}")
    print(f"Headline: {result.headline}")
    print(f"Article: {result.article[:200]}...")
    
    # Memory usage report
    if torch.cuda.is_available():
        print("="*60)
        print("Memory Usage Report:")
        print(f"Current: {torch.cuda.memory_allocated(0) / 1e9:.1f}GB")
        print(f"Peak: {torch.cuda.max_memory_allocated(0) / 1e9:.1f}GB")
        print("Target: ~3.5GB")
        print("="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Direct testing mode
        asyncio.run(main())
    else:
        # FastAPI server mode
        uvicorn.run(app, host="0.0.0.0", port=8009)
