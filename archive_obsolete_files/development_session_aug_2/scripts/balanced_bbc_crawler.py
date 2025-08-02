#!/usr/bin/env python3
"""
Balanced BBC NewsReader Crawler - Crash Prevention + Performance

This implementation balances crash safety with performance:
- Keeps model loaded but monitors memory carefully
- Moderate cleanup between articles
- Emergency shutdown triggers maintained
- Optimized for 19-article processing
"""

import asyncio
import json
import logging
import time
import gc
import psutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Import our crash-safe NewsReader
import sys
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
logger = logging.getLogger("balanced_crawler")

class BalancedBBCCrawler:
    """
    Balanced BBC Crawler - Performance + Safety
    """
    
    def __init__(self, max_memory_gb: float = 16.0):
        self.newsreader = None
        self.browser = None
        self.context = None
        self.results = []
        self.failed_urls = []
        self.processed_count = 0
        self.max_memory_gb = max_memory_gb
        
        # Moderate cleanup thresholds
        self.memory_warning_threshold = 0.8  # 80% of limit
        self.memory_emergency_threshold = 0.9  # 90% of limit
    
    def check_memory_status(self, operation: str) -> str:
        """Check memory status and return action needed"""
        try:
            # Check system memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Check GPU memory
            gpu_percent = 0
            if self.newsreader:
                import torch
                if torch.cuda.is_available():
                    current_gpu = torch.cuda.memory_allocated()
                    max_gpu = self.max_memory_gb * 1024**3
                    gpu_percent = (current_gpu / max_gpu) * 100
            
            logger.info(f"📊 {operation}: System {memory_percent:.1f}%, GPU {gpu_percent:.1f}%")
            
            # Determine action
            if memory_percent > 90 or gpu_percent > self.memory_emergency_threshold * 100:
                return "emergency"
            elif memory_percent > 80 or gpu_percent > self.memory_warning_threshold * 100:
                return "cleanup"
            else:
                return "continue"
                
        except Exception as e:
            logger.error(f"❌ Memory check failed: {e}")
            return "cleanup"
    
    async def moderate_cleanup(self):
        """Moderate cleanup - clear cache but keep model"""
        try:
            logger.info("🧹 Moderate cleanup...")
            
            # Python garbage collection
            gc.collect()
            
            # CUDA cache cleanup (but keep model loaded)
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ Moderate cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Moderate cleanup failed: {e}")
    
    async def initialize_balanced(self) -> bool:
        """Balanced initialization"""
        try:
            logger.info("⚖️ Starting balanced initialization...")
            
            # Initialize NewsReader with balanced limits
            self.newsreader = CrashSafeNewsReader(max_gpu_memory_gb=self.max_memory_gb)
            
            if not await self.newsreader.initialize_with_memory_safety():
                logger.error("❌ NewsReader initialization failed")
                return False
            
            # Initialize browser with moderate settings
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(
                headless=True,
                args=['--disable-dev-shm-usage', '--no-sandbox']
            )
            
            self.context = await self.browser.new_context(
                viewport={'width': 1200, 'height': 800},
                user_agent='Mozilla/5.0 (compatible; BalancedCrawler/1.0)'
            )
            
            logger.info("✅ Balanced initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Balanced initialization failed: {e}")
            return False
    
    async def get_bbc_england_urls(self, max_urls: int = 19) -> List[str]:
        """Get BBC England URLs"""
        try:
            logger.info(f"🔍 Getting {max_urls} BBC England URLs...")
            
            page = await self.context.new_page()
            
            try:
                await page.goto("https://www.bbc.co.uk/news/england", 
                              wait_until='networkidle', timeout=30000)
                await asyncio.sleep(2)
                
                urls = set()
                
                # Look for article links
                links = await page.locator('a[href*=\"articles/\"]').all()
                
                for link in links[:max_urls*2]:  # Get extra in case some fail
                    try:
                        href = await link.get_attribute('href')
                        if href:
                            if href.startswith('/'):
                                href = f"https://www.bbc.co.uk{href}"
                            
                            if 'articles/' in href:
                                urls.add(href)
                                
                            if len(urls) >= max_urls:
                                break
                    except Exception:
                        continue
                
                england_urls = list(urls)[:max_urls]
                logger.info(f"✅ Found {len(england_urls)} URLs")
                return england_urls
                
            finally:
                await page.close()
                
        except Exception as e:
            logger.error(f"❌ URL discovery failed: {e}")
            return []
    
    async def process_article_balanced(self, url: str, article_num: int, total: int) -> Optional[Dict]:
        """Process article with balanced approach"""
        try:
            logger.info(f"📰 [{article_num}/{total}] Processing: {url}")
            
            # Check memory before processing
            memory_status = self.check_memory_status(f"Article {article_num}")
            
            if memory_status == "emergency":
                logger.error(f"🚨 Emergency memory level, skipping article {article_num}")
                return None
            elif memory_status == "cleanup":
                await self.moderate_cleanup()
            
            # Create page
            page = await self.context.new_page()
            
            try:
                # Navigate
                await page.goto(url, wait_until='networkidle', timeout=45000)
                await asyncio.sleep(2)
                
                # Handle cookies
                try:
                    cookie_selectors = ['button:has-text("Accept")', 'button:has-text("OK")']
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
                    pass
                
                # Get title
                title = await page.title()
                
                # Screenshot
                screenshot_bytes = await page.screenshot(
                    full_page=False,
                    type='png'
                )
                
                image = Image.open(io.BytesIO(screenshot_bytes))
                
                # Moderate resize for balance
                max_size = 1000
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                logger.info(f"📸 Screenshot: {image.size}")
                
                # Analyze with context
                context = f"This is BBC England news article titled '{title}'. Please extract the main headline, key details, important facts, and news content from this webpage."
                
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
                    
                    logger.info(f"✅ Article {article_num} completed successfully")
                    
                    # Light cleanup every few articles
                    if article_num % 5 == 0:
                        await self.moderate_cleanup()
                    
                    return result
                else:
                    logger.warning(f"⚠️ No analysis for article {article_num}")
                    return None
                    
            finally:
                await page.close()
                
        except Exception as e:
            logger.error(f"❌ Article {article_num} failed: {e}")
            return None
    
    async def crawl_bbc_england_balanced(self, max_articles: int = 19) -> Dict:
        """Balanced BBC England crawling"""
        try:
            start_time = time.time()
            
            logger.info(f"⚖️ Starting balanced BBC England crawling ({max_articles} articles)")
            
            # Get URLs
            urls = await self.get_bbc_england_urls(max_articles)
            if not urls:
                return {"error": "No URLs found", "results": []}
            
            logger.info(f"📝 Processing {len(urls)} articles...")
            
            # Process articles
            for i, url in enumerate(urls, 1):
                # Small delay between articles
                if i > 1:
                    await asyncio.sleep(1)
                
                result = await self.process_article_balanced(url, i, len(urls))
                if result:
                    self.results.append(result)
                    self.processed_count += 1
                else:
                    self.failed_urls.append({
                        "url": url, 
                        "article_number": i, 
                        "reason": "processing_failed"
                    })
            
            # Final results
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            
            crawl_results = {
                "success": True,
                "balanced_mode": True,
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
            
            logger.info(f"✅ Balanced crawling complete: {len(self.results)}/{len(urls)} successful")
            return crawl_results
            
        except Exception as e:
            logger.error(f"❌ Balanced crawling failed: {e}")
            return {"error": str(e), "results": self.results}
    
    async def cleanup(self):
        """Final cleanup"""
        try:
            logger.info("🧹 Final cleanup...")
            
            if self.context:
                await self.context.close()
            
            if self.browser:
                await self.browser.close()
            
            if self.newsreader:
                self.newsreader.cleanup()
            
            logger.info("✅ Final cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

async def main():
    """Balanced BBC crawler execution"""
    
    crawler = BalancedBBCCrawler(max_memory_gb=16.0)
    
    try:
        # Initialize
        if not await crawler.initialize_balanced():
            logger.error("❌ Failed to initialize balanced crawler")
            return
        
        # Run crawling
        results = await crawler.crawl_bbc_england_balanced(max_articles=19)
        
        # Save results
        output_file = "bbc_england_balanced_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Display results
        print("\n" + "="*80)
        print("⚖️ BALANCED BBC ENGLAND CRAWLER RESULTS")
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
        
        # Show analyses
        if results.get('results'):
            print("\n📰 SUCCESSFUL NEWS ANALYSES:")
            print("-" * 80)
            for i, result in enumerate(results['results'][:5], 1):
                print(f"{i}. Title: {result.get('title', 'N/A')}")
                print(f"   Analysis: {result['analysis'][:200]}...")
                print()
        
    finally:
        await crawler.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
