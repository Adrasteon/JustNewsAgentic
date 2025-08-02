#!/usr/bin/env python3
"""
Process-Isolated BBC NewsReader Crawler

This approach uses subprocess isolation to prevent system crashes:
- Each article is processed in a separate Python process
- If a process crashes, it doesn't affect the main system
- Memory is completely cleared between articles
- Maximum isolation for system stability
"""

import asyncio
import json
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("isolated_bbc_crawler")

class ProcessIsolatedBBCCrawler:
    """
    BBC Crawler using process isolation for maximum stability
    """
    
    def __init__(self):
        self.results = []
        self.failed_urls = []
        
    async def get_bbc_england_urls(self, max_urls: int = 5) -> list:
        """Get BBC England article URLs with conservative limit"""
        try:
            logger.info("🔍 Discovering BBC England article URLs...")
            
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            await page.goto("https://www.bbc.co.uk/news/england", wait_until='networkidle')
            await asyncio.sleep(3)
            
            # Get article links
            links = await page.locator('a[href*="articles/"]').all()
            urls = set()
            
            for link in links[:max_urls*2]:  # Get extra in case some fail
                try:
                    href = await link.get_attribute('href')
                    if href:
                        if href.startswith('/'):
                            href = f"https://www.bbc.co.uk{href}"
                        if 'articles/' in href and 'sport' not in href:
                            urls.add(href)
                except:
                    continue
            
            await browser.close()
            
            url_list = list(urls)[:max_urls]
            logger.info(f"✅ Found {len(url_list)} BBC England URLs")
            return url_list
            
        except Exception as e:
            logger.error(f"❌ URL discovery failed: {e}")
            return []
    
    def process_single_article(self, url: str, article_num: int, total: int) -> dict:
        """
        Process a single article in complete isolation.
        This method runs in a separate Python process.
        """
        
        # Create isolated processing script
        process_script = f'''
import asyncio
import sys
import gc
import torch
import warnings
warnings.filterwarnings("ignore")

async def process_article():
    try:
        sys.path.append('/home/adra/JustNewsAgentic')
        from crash_safe_newsreader import CrashSafeNewsReader
        from playwright.async_api import async_playwright
        from PIL import Image
        import io
        import json
        
        print(f"🔄 [{article_num}/{total}] Processing: {{url}}", flush=True)
        
        # Initialize NewsReader with very conservative settings
        newsreader = CrashSafeNewsReader(max_gpu_memory_gb=12.0)  # Very conservative
        
        # Initialize with timeout
        init_success = await asyncio.wait_for(
            newsreader.initialize_with_memory_safety(),
            timeout=120  # 2 minute timeout
        )
        
        if not init_success:
            return {{"error": "NewsReader initialization failed", "url": "{url}"}}
        
        # Capture screenshot with timeout
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={{'width': 1280, 'height': 720}}  # Smaller viewport
        )
        page = await context.new_page()
        
        await asyncio.wait_for(
            page.goto("{url}", wait_until='networkidle'),
            timeout=30
        )
        await asyncio.sleep(2)
        
        # Get page title
        title = await page.title()
        
        # Capture smaller screenshot
        screenshot_bytes = await page.screenshot(
            full_page=False,  # Only visible area
            type='png'
        )
        
        await browser.close()
        
        # Analyze with NewsReader
        image = Image.open(io.BytesIO(screenshot_bytes))
        
        # Resize for memory efficiency
        if max(image.size) > 800:
            image.thumbnail((800, 600), Image.Resampling.LANCZOS)
        
        analysis = await asyncio.wait_for(
            newsreader.safe_analyze_content(
                image, 
                "Extract the main headline and key points from this BBC news article."
            ),
            timeout=60  # 1 minute timeout
        )
        
        # Clean up immediately
        newsreader.cleanup()
        del newsreader
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        result = {{
            "url": "{url}",
            "title": title,
            "analysis": analysis,
            "timestamp": "{{datetime.now().isoformat()}}",
            "status": "success",
            "article_number": {article_num}
        }}
        
        print(f"✅ Article {{article_num}} completed", flush=True)
        return result
        
    except Exception as e:
        print(f"❌ Article {{article_num}} failed: {{e}}", flush=True)
        return {{
            "url": "{url}",
            "error": str(e),
            "status": "failed",
            "article_number": {article_num}
        }}

# Run the processing
import asyncio
from datetime import datetime
result = asyncio.run(process_article())
print("RESULT_JSON:" + json.dumps(result), flush=True)
'''
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(process_script)
            script_path = f.name
        
        try:
            # Run in completely isolated process with resource limits
            process = subprocess.Popen([
                '/home/adra/miniconda3/envs/rapids-25.06/bin/python', script_path
            ], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            env={'CUDA_VISIBLE_DEVICES': '0'}  # Ensure single GPU
            )
            
            # Wait with timeout
            stdout, _ = process.communicate(timeout=300)  # 5 minute timeout
            
            # Extract result from output
            for line in stdout.split('\n'):
                if line.startswith('RESULT_JSON:'):
                    result_json = line[12:]  # Remove "RESULT_JSON:" prefix
                    return json.loads(result_json)
            
            return {"error": "No result found in process output", "url": url}
            
        except subprocess.TimeoutExpired:
            process.kill()
            return {"error": "Process timeout", "url": url}
        except Exception as e:
            return {"error": f"Process execution failed: {e}", "url": url}
        finally:
            # Clean up script file
            try:
                Path(script_path).unlink()
            except:
                pass
    
    async def run_isolated_crawl(self, max_articles: int = 5):
        """Run the isolated crawling process"""
        start_time = time.time()
        
        logger.info(f"🚀 Starting Process-Isolated BBC Crawler (max {max_articles} articles)")
        
        # Get URLs
        urls = await self.get_bbc_england_urls(max_articles)
        if not urls:
            return {"error": "No URLs found"}
        
        logger.info(f"📝 Processing {len(urls)} articles with process isolation...")
        
        # Process each article in complete isolation
        for i, url in enumerate(urls, 1):
            logger.info(f"🔄 Starting article {i}/{len(urls)} in isolated process...")
            
            # Add delay between articles to let system recover
            if i > 1:
                logger.info("⏳ Waiting 10 seconds between articles...")
                await asyncio.sleep(10)
            
            try:
                # Process in isolated subprocess
                result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.process_single_article, 
                    url, i, len(urls)
                )
                
                if result.get('status') == 'success':
                    self.results.append(result)
                    logger.info(f"✅ Article {i} processed successfully")
                else:
                    self.failed_urls.append(result)
                    logger.warning(f"⚠️ Article {i} failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                logger.error(f"❌ Article {i} processing failed: {e}")
                self.failed_urls.append({"url": url, "error": str(e)})
        
        # Compile results
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        final_results = {
            "success": True,
            "total_urls": len(urls),
            "successful_analyses": len(self.results),
            "failed_analyses": len(self.failed_urls),
            "success_rate": round((len(self.results) / len(urls) * 100), 1) if urls else 0,
            "processing_time_seconds": duration,
            "isolation_method": "subprocess",
            "results": self.results,
            "failed_urls": self.failed_urls,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        output_file = "bbc_england_isolated_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Isolated crawling complete: {len(self.results)}/{len(urls)} successful")
        return final_results

async def main():
    """Run the process-isolated BBC crawler"""
    crawler = ProcessIsolatedBBCCrawler()
    
    try:
        results = await crawler.run_isolated_crawl(max_articles=5)  # Start with just 5
        
        print("\n" + "="*80)
        print("🏆 PROCESS-ISOLATED BBC CRAWLER RESULTS")
        print("="*80)
        print(f"📊 Success Rate: {results.get('success_rate', 0)}%")
        print(f"✅ Successful: {results.get('successful_analyses', 0)}")
        print(f"❌ Failed: {results.get('failed_analyses', 0)}")
        print(f"⏱️  Duration: {results.get('processing_time_seconds', 0)}s")
        print(f"🔒 Method: Process Isolation")
        print("="*80)
        
        # Show sample results
        if results.get('results'):
            print("\n📰 SAMPLE ANALYSIS:")
            sample = results['results'][0]
            print(f"Title: {sample.get('title', 'N/A')}")
            print(f"Analysis: {sample.get('analysis', 'N/A')[:200]}...")
        
    except Exception as e:
        logger.error(f"❌ Isolated crawling failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
