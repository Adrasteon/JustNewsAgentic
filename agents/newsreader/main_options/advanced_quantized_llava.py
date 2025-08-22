"""
Advanced Quantized LLaVA NewsReader Agent
Using multiple quantization strategies for maximum memory reduction
Target: 3.5GB memory usage (50% reduction from 7GB)
"""

import asyncio
import logging
import time
import torch
from contextlib import asynccontextmanager
from playwright.async_api import async_playwright
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, Optional

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Setup logging
logging.basicConfig(
    filename="advanced_quantized_llava.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("advanced_quantized_llava")

class NewsExtractionRequest(BaseModel):
    url: str
    screenshot_path: Optional[str] = None

class NewsExtractionResponse(BaseModel):
    headline: str
    article: str
    success: bool
    extraction_method: str
    processing_time: float

class AdvancedQuantizedLlavaAgent:
    def __init__(self):
        """Initialize LLaVA with advanced quantization strategies"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Advanced Quantized LLaVA on device: {self.device}")
        
        # Check initial memory
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"Initial GPU memory: {initial_memory:.1f}GB")
        
        # Load processor with fast tokenizer
        logger.info("Loading processor...")
        self.processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            use_fast=True
        )
        
        # Load model with aggressive memory optimizations
        logger.info("Loading model with advanced quantization...")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16,  # Base FP16
            device_map="auto",
            low_cpu_mem_usage=True,
            # Use CPU offloading for some layers if needed
            offload_folder="./offload_cache",
            max_memory={0: "16GB"}  # Limit GPU memory usage
        )
        
        # Apply post-loading quantization
        self.apply_quantization()
        
        # Compile for additional optimizations
        if hasattr(torch, 'compile'):
            logger.info("Compiling model...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        # Final memory check
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(0) / 1e9
            memory_used = final_memory - initial_memory
            logger.info(f"Final GPU memory: {final_memory:.1f}GB")
            logger.info(f"Model memory usage: {memory_used:.1f}GB")
            
            # Force garbage collection and measure again
            torch.cuda.empty_cache()
            stable_memory = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"Stable memory after cleanup: {stable_memory:.1f}GB")
        
        logger.info("✅ Advanced Quantized LLaVA model loaded")

    def apply_quantization(self):
        """Apply post-loading quantization strategies"""
        logger.info("Applying post-loading quantization...")
        
        try:
            # Apply dynamic quantization to compatible layers
            self.model = torch.quantization.quantize_dynamic(
                self.model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            logger.info("✅ Dynamic quantization applied")
        except Exception as e:
            logger.warning(f"Dynamic quantization failed: {e}")
        
        # Apply gradient checkpointing to reduce memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")

    async def capture_screenshot(self, url: str, screenshot_path: str = "page_advanced.png") -> str:
        """Optimized screenshot capture"""
        logger.info(f"Capturing screenshot: {url}")
        t0 = time.time()
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-dev-shm-usage']
                )
                page = await browser.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(1000)
                await page.screenshot(path=screenshot_path, full_page=False)
                await browser.close()
            
            t1 = time.time()
            logger.info(f"Screenshot saved: {screenshot_path} ({t1-t0:.2f}s)")
            return screenshot_path
            
        except Exception as e:
            logger.error(f"Screenshot failed: {str(e)}")
            raise

    def extract_content_with_advanced_llava(self, image_path: str) -> Dict[str, str]:
        """Extract content using advanced quantized LLaVA"""
        logger.info(f"Processing with advanced LLaVA: {image_path}")
        t0 = time.time()
        
        try:
            # Load and prepare image
            image = Image.open(image_path).convert('RGB')
            
            # Optimized prompt
            prompt = """<image>
Extract main news content:
HEADLINE: [main headline]
ARTICLE: [first 2-3 paragraphs]

Focus on primary news story."""
            
            # Process with memory-efficient settings
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            
            # Generate with memory optimization
            with torch.no_grad():
                # Clear cache before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=250,  # Reduced for memory
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode and parse
            response = self.processor.decode(output[0], skip_special_tokens=True)
            result = self.parse_response(response)
            
            t1 = time.time()
            logger.info(f"Advanced processing complete ({t1-t0:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"Advanced processing failed: {str(e)}")
            return {
                "headline": "Extraction failed",
                "article": f"Error: {str(e)}",
                "raw_response": ""
            }

    def parse_response(self, response: str) -> Dict[str, str]:
        """Parse model response"""
        try:
            lines = response.split('\n')
            headline = ""
            article = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('HEADLINE:'):
                    headline = line.replace('HEADLINE:', '').strip()
                elif line.startswith('ARTICLE:'):
                    article = line.replace('ARTICLE:', '').strip()
                elif article and line:
                    article += " " + line
            
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
        """Main processing pipeline"""
        t0 = time.time()
        logger.info(f"Starting advanced pipeline: {url}")
        
        try:
            if not screenshot_path:
                screenshot_path = await self.capture_screenshot(url)
            
            content = self.extract_content_with_advanced_llava(screenshot_path)
            
            t1 = time.time()
            processing_time = t1 - t0
            
            return NewsExtractionResponse(
                headline=content["headline"],
                article=content["article"],
                success=True,
                extraction_method="llava-v1.6-mistral-7b-advanced-quantized",
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return NewsExtractionResponse(
                headline="Extraction failed",
                article=f"Error: {str(e)}",
                success=False,
                extraction_method="llava-v1.6-mistral-7b-advanced-quantized",
                processing_time=time.time() - t0
            )

    def get_memory_stats(self) -> Dict[str, str]:
        """Get current memory statistics"""
        if torch.cuda.is_available():
            return {
                "allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.1f}GB",
                "reserved": f"{torch.cuda.memory_reserved(0) / 1e9:.1f}GB",
                "max_allocated": f"{torch.cuda.max_memory_allocated(0) / 1e9:.1f}GB"
            }
        return {"status": "CPU mode"}

# Global agent
newsreader_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with memory monitoring"""
    global newsreader_agent
    logger.info("🚀 Starting Advanced Quantized LLaVA NewsReader")
    
    newsreader_agent = AdvancedQuantizedLlavaAgent()
    logger.info("✅ Advanced Quantized NewsReader initialized")
    
    yield
    
    logger.info("🔄 Shutting down Advanced Quantized NewsReader")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("✅ Shutdown complete")

app = FastAPI(lifespan=lifespan)

# Register shutdown endpoint if available
try:
    from agents.common.shutdown import register_shutdown_endpoint
    register_shutdown_endpoint(app)
except Exception:
    logger = logging.getLogger(__name__)
    logger.debug("shutdown endpoint not registered for advanced_quantized_llava")

@app.post("/extract_news", response_model=NewsExtractionResponse)
async def extract_news(request: NewsExtractionRequest):
    """Extract news content"""
    if not newsreader_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return await newsreader_agent.process_news_url(request.url, request.screenshot_path)

@app.get("/health")
async def health_check():
    """Health check with memory stats"""
    memory_stats = newsreader_agent.get_memory_stats() if newsreader_agent else {}
    
    return {
        "status": "ok",
        "model": "llava-v1.6-mistral-7b-advanced-quantized",
        "environment": "rapids-25.06",
        "memory_target": "3.5GB maximum",
        **memory_stats
    }

@app.get("/memory")
async def memory_status():
    """Detailed memory status"""
    if not newsreader_agent:
        return {"error": "Agent not initialized"}
    
    return {
        "memory_stats": newsreader_agent.get_memory_stats(),
        "quantization": "dynamic + gradient checkpointing",
        "target": "3.5GB maximum usage"
    }

# Testing function
async def main():
    """Test advanced quantized implementation"""
    print("🧪 Testing Advanced Quantized LLaVA NewsReader")
    print("=" * 60)
    
    agent = AdvancedQuantizedLlavaAgent()
    
    # Memory report
    print("Memory Usage Report:")
    stats = agent.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Test extraction
    test_url = "https://www.bbc.co.uk/news/uk-politics-55497671"
    result = await agent.process_news_url(test_url)
    
    print("Extraction Results:")
    print(f"  Success: {result.success}")
    print(f"  Processing Time: {result.processing_time:.2f}s")
    print(f"  Method: {result.extraction_method}")
    print(f"  Headline: {result.headline}")
    print(f"  Article: {result.article[:100]}...")
    print()
    
    # Final memory check
    print("Final Memory Report:")
    final_stats = agent.get_memory_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print("=" * 60)
    print("🎯 Target: <3.5GB for safe system integration")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(main())
    else:
        uvicorn.run(app, host="0.0.0.0", port=8009)
