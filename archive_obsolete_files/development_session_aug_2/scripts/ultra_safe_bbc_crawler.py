#!/usr/bin/env python3
"""
Ultra-Safe BBC NewsReader Crawler - Crash Prevention Focus

This implementation prioritizes system stability over speed:
- Single article processing with full cleanup between articles
- Aggressive memory monitoring and limits
- Process isolation for crash safety
- Emergency shutdown triggers
- Minimal memory footprint mode
"""

import asyncio
import json
import logging
import time
import gc
import psutil
import signal
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Import our crash-safe NewsReader
sys.path.append('/home/adra/JustNewsAgentic')
from crash_safe_newsreader import CrashSafeNewsReader
from playwright.async_api import async_playwright, Page, Browser
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ultra_safe_crawler")

class UltraSafeBBCCrawler:
    """
    Ultra-Safe BBC Crawler with Crash Prevention Focus
    
    Features:
    - Single article processing with full cleanup
    - Aggressive memory monitoring
    - Emergency shutdown triggers
    - Process isolation
    - Minimal memory footprint
    """
    
    def __init__(self, max_memory_gb: float = 15.0):
        self.newsreader = None
        self.browser = None
        self.context = None
        self.results = []
        self.failed_urls = []
        self.processed_count = 0
        self.max_memory_gb = max_memory_gb
        self.emergency_shutdown = False
        
        # Memory monitoring thresholds
        self.memory_warning_threshold = 0.7  # 70% of limit
        self.memory_emergency_threshold = 0.85  # 85% of limit
        
        # Set up emergency shutdown handler
        signal.signal(signal.SIGTERM, self._emergency_shutdown_handler)
        signal.signal(signal.SIGINT, self._emergency_shutdown_handler)
    
    def _emergency_shutdown_handler(self, signum, frame):
        """Emergency shutdown handler"""
        logger.error(f"🚨 Emergency shutdown triggered by signal {signum}")
        self.emergency_shutdown = True
        asyncio.create_task(self.emergency_cleanup())
        sys.exit(1)
    
    def check_memory_safety(self, operation: str) -> bool:
        """Check if it's safe to continue based on memory usage"""
        try:
            # Check system memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Check GPU memory if available
            gpu_percent = 0
            if hasattr(self, 'newsreader') and self.newsreader:
                import torch
                if torch.cuda.is_available():
                    current_gpu = torch.cuda.memory_allocated()
                    max_gpu = self.max_memory_gb * 1024**3
                    gpu_percent = (current_gpu / max_gpu) * 100
            
            logger.info(f"🔍 Memory check for {operation}: System {memory_percent:.1f}%, GPU {gpu_percent:.1f}%")
            
            # Emergency threshold
            if memory_percent > 85 or gpu_percent > self.memory_emergency_threshold * 100:
                logger.error(f"🚨 EMERGENCY: Memory usage too high - System: {memory_percent:.1f}%, GPU: {gpu_percent:.1f}%")
                self.emergency_shutdown = True
                return False
            
            # Warning threshold
            if memory_percent > 75 or gpu_percent > self.memory_warning_threshold * 100:
                logger.warning(f"⚠️ HIGH MEMORY: System: {memory_percent:.1f}%, GPU: {gpu_percent:.1f}%")
                # Force cleanup but continue
                asyncio.create_task(self.force_cleanup())
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory check failed: {e}")
            return False
    
    async def force_cleanup(self):
        """Force aggressive cleanup"""
        try:
            logger.info("🧹 Forcing aggressive cleanup...")
            
            # Python garbage collection
            gc.collect()
            
            # NewsReader cleanup
            if self.newsreader:
                self.newsreader.emergency_cleanup()
            
            # CUDA cleanup
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("✅ Aggressive cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Force cleanup failed: {e}")
    
    async def initialize_ultra_safe(self) -> bool:
        """Ultra-safe initialization with minimal memory footprint"""
        try:
            logger.info("🛡️ Starting ultra-safe initialization...")
            
            if not self.check_memory_safety("initialization"):
                return False
            
            # Initialize NewsReader with conservative limits
            self.newsreader = CrashSafeNewsReader(max_gpu_memory_gb=self.max_memory_gb)
            
            if not await self.newsreader.initialize_with_memory_safety():
                logger.error("❌ NewsReader initialization failed")
                return False
            
            # Initialize minimal browser
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(
                headless=True,
                args=[
                    '--disable-dev-shm-usage',  # Reduce shared memory usage
                    '--disable-gpu',  # Disable GPU for browser
                    '--no-sandbox',  # Reduce memory overhead
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            )
            
            self.context = await self.browser.new_context(
                viewport={'width': 1024, 'height': 768},  # Smaller viewport
                user_agent='Mozilla/5.0 (compatible; SafeCrawler/1.0)'
            )
            
            logger.info("✅ Ultra-safe initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ultra-safe initialization failed: {e}")
            await self.emergency_cleanup()
            return False
    
    async def get_bbc_england_urls(self, max_urls: int = 19) -> List[str]:
        """Get BBC England URLs with memory safety"""
        try:
            if not self.check_memory_safety("URL discovery"):
                return []
            
            logger.info(f"🔍 Discovering up to {max_urls} BBC England article URLs...")
            
            page = await self.context.new_page()
            
            try:
                await page.goto("https://www.bbc.co.uk/news/england", 
                              wait_until='networkidle', timeout=30000)
                await asyncio.sleep(2)
                
                urls = set()
                
                # Look for article links
                selectors = ['a[href*="articles/"]']
                
                for selector in selectors:
                    try:
                        links = await page.locator(selector).all()
                        
                        for link in links[:max_urls]:  # Limit processing
                            try:
                                href = await link.get_attribute('href')
                                if href:
                                    if href.startswith('/'):
                                        href = f"https://www.bbc.co.uk{href}"
                                    
                                    # Filter for news articles
                                    if 'articles/' in href:
                                        urls.add(href)
                                        
                                    if len(urls) >= max_urls:
                                        break
                            except Exception:
                                continue
                    except Exception as e:
                        logger.warning(f"⚠️ Selector {selector} failed: {e}")
                        continue
                
                england_urls = list(urls)[:max_urls]
                logger.info(f"✅ Found {len(england_urls)} BBC England URLs")
                return england_urls
                
            finally:
                await page.close()
                
        except Exception as e:
            logger.error(f"❌ URL discovery failed: {e}")
            return []
    
    async def process_single_article_ultra_safe(self, url: str, article_num: int, total: int) -> Optional[Dict]:
        """Process single article with maximum safety"""
        try:
            logger.info(f"🔄 [{article_num}/{total}] Processing: {url}")
            
            # Memory check before processing
            if not self.check_memory_safety(f"article {article_num}"):
                logger.error(f"❌ Skipping article {article_num} due to memory constraints")
                return None
            
            if self.emergency_shutdown:
                logger.error("🚨 Emergency shutdown detected, stopping processing")
                return None
            
            # Create new page for each article (isolated processing)
            page = await self.context.new_page()
            
            try:
                # Navigate with timeout
                await page.goto(url, wait_until='networkidle', timeout=45000)
                await asyncio.sleep(3)
                
                # Handle cookie banners
                try:
                    cookie_selectors = [
                        'button:has-text("Accept")',
                        'button:has-text("OK")',
                        'button:has-text("Agree")',
                        '[data-testid="cookie-accept-all"]'
                    ]
                    for selector in cookie_selectors:
                        try:
                            button = page.locator(selector)
                            if await button.count() > 0:
                                await button.first.click()
                                await asyncio.sleep(1)
                                break
                        except:
                            continue
                except:
                    pass  # Cookie handling is optional
                
                # Get page title for context
                title = await page.title()
                
                # Capture screenshot with safety limits
                screenshot_bytes = await page.screenshot(
                    full_page=False,  # Only visible area for memory safety
                    type='png'
                )
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(screenshot_bytes))
                
                # Resize if too large
                max_size = 800  # Conservative size
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                logger.info(f"✅ Screenshot captured and resized: {image.size}")
                
                # Analyze with NewsReader
                context = f"This is a BBC England news article with title: '{title}'. Extract the main headline, key details, and news content from this webpage screenshot."
                
                analysis = await self.newsreader.safe_analyze_content(image, context)
                
                if analysis:
                    result = {
                        "article_number": article_num,
                        "url": url,
                        "title": title,
                        "analysis": analysis,
                        "timestamp": datetime.now().isoformat(),
                        "image_size": f"{image.size[0]}x{image.size[1]}",
                        "status": "success"
                    }
                    
                    logger.info(f"✅ Article {article_num} analysis complete")
                    
                    # Force cleanup after each article
                    await self.force_cleanup()
                    
                    return result
                else:
                    logger.warning(f"⚠️ No analysis result for article {article_num}")
                    return None
                    
            finally:
                # Always close page
                await page.close()
                
        except Exception as e:
            logger.error(f"❌ Article {article_num} processing failed: {e}")
            await self.force_cleanup()
            return None
    
    async def crawl_bbc_england_ultra_safe(self, max_articles: int = 19) -> Dict:
        """Ultra-safe BBC England crawling"""
        try:
            start_time = time.time()
            
            logger.info(f"🛡️ Starting ultra-safe BBC England crawling (max {max_articles} articles)")
            
            # Get URLs
            urls = await self.get_bbc_england_urls(max_articles)
            if not urls:
                return {"error": "No URLs found", "results": []}
            
            logger.info(f"📝 Processing {len(urls)} articles with ultra-safe mode...")
            
            # Process articles one by one with safety checks
            for i, url in enumerate(urls, 1):
                if self.emergency_shutdown:
                    logger.error("🚨 Emergency shutdown, stopping crawl")
                    break
                
                # Safety delay between articles
                if i > 1:
                    await asyncio.sleep(2)
                
                result = await self.process_single_article_ultra_safe(url, i, len(urls))
                if result:
                    self.results.append(result)
                    self.processed_count += 1
                else:
                    self.failed_urls.append({"url": url, "article_number": i, "reason": "processing_failed"})
                
                # Memory check after each article
                if not self.check_memory_safety(f"after article {i}"):
                    logger.error(f"🚨 Memory safety check failed after article {i}, stopping crawl")
                    break
            
            # Compile results
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            
            crawl_results = {
                "success": True,
                "ultra_safe_mode": True,
                "total_urls_found": len(urls),
                "successful_analyses": len(self.results),
                "failed_analyses": len(self.failed_urls),
                "success_rate": round((len(self.results) / len(urls) * 100), 1) if urls else 0,
                "processing_time_seconds": duration,
                "average_time_per_article": round(duration / len(self.results), 2) if self.results else 0,
                "newsreader_model": self.newsreader.model_type if self.newsreader else "unknown",
                "memory_limit_gb": self.max_memory_gb,
                "results": self.results,
                "failed_urls": self.failed_urls,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Ultra-safe crawling complete: {len(self.results)}/{len(urls)} successful")
            return crawl_results
            
        except Exception as e:
            logger.error(f"❌ Ultra-safe crawling failed: {e}")
            return {"error": str(e), "results": self.results}
    
    async def emergency_cleanup(self):
        """Emergency cleanup for crash prevention"""
        try:
            logger.warning("🚨 Emergency cleanup initiated...")
            
            if self.context:
                await self.context.close()
            
            if self.browser:
                await self.browser.close()
            
            if self.newsreader:
                self.newsreader.cleanup()
            
            # Force memory cleanup
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ Emergency cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Emergency cleanup error: {e}")

async def main():
    """Ultra-safe BBC crawler execution"""
    
    crawler = UltraSafeBBCCrawler(max_memory_gb=15.0)  # Conservative 15GB limit
    
    try:
        # Initialize ultra-safe mode
        if not await crawler.initialize_ultra_safe():
            logger.error("❌ Failed to initialize ultra-safe crawler")
            return
        
        # Run BBC England crawling
        results = await crawler.crawl_bbc_england_ultra_safe(max_articles=19)
        
        # Save results
        output_file = "bbc_england_ultra_safe_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Display comprehensive summary
        print("\n" + "="*80)
        print("🛡️ ULTRA-SAFE BBC ENGLAND CRAWLER RESULTS")
        print("="*80)
        print(f"📊 Success Rate: {results.get('success_rate', 0)}%")
        print(f"✅ Successful: {results.get('successful_analyses', 0)}")
        print(f"❌ Failed: {results.get('failed_analyses', 0)}")
        print(f"⏱️  Total Duration: {results.get('processing_time_seconds', 0)}s")
        print(f"📈 Avg per Article: {results.get('average_time_per_article', 0)}s")
        print(f"🤖 Model: {results.get('newsreader_model', 'unknown')}")
        print(f"💾 Memory Limit: {results.get('memory_limit_gb', 0)}GB")
        print(f"📄 Results: {output_file}")
        print("="*80)
        
        # Show sample results
        if results.get('results'):
            print("\n📰 SAMPLE ANALYSES:")
            print("-" * 80)
            for i, result in enumerate(results['results'][:3], 1):
                print(f"{i}. Title: {result.get('title', 'N/A')}")
                print(f"   URL: {result['url']}")
                print(f"   Analysis: {result['analysis'][:150]}...")
                print()
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")
    finally:
        # Always cleanup
        await crawler.emergency_cleanup()

if __name__ == "__main__":
    asyncio.run(main())
