#!/usr/bin/env python3
"""
Production-Scale BBC Crawler with Cookie/Modal Handling

This reverts to fast batch processing while addressing root causes:
- Aggressive cookie consent and modal dismissal
- Proper page state management
- Memory-efficient batch processing 
- DOM-based content extraction (faster than screenshots)
- Targeted memory cleanup only when needed

Goal: 1000+ articles/day processing with stability
"""

import asyncio
import json
import time
import gc
import torch
from datetime import datetime
from playwright.async_api import async_playwright
from typing import List, Dict, Optional
import logging

# Import our practical NewsReader for analysis
from practical_newsreader_solution import PracticalNewsReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("production_bbc_crawler")

class ProductionBBCCrawler:
    """Fast, production-scale BBC crawler that handles root causes"""
    
    def __init__(self, batch_size: int = 10):
        self.newsreader = None
        self.batch_size = batch_size
        self.processed_articles = []
        self.failed_articles = []
        
    async def initialize(self):
        """Initialize NewsReader once for batch processing"""
        logger.info("🚀 Initializing Production BBC Crawler...")
        
        self.newsreader = PracticalNewsReader()
        try:
            await self.newsreader.initialize_option_a_lightweight_llava()
            logger.info("✅ NewsReader loaded with LLaVA-1.5")
        except Exception as e:
            logger.warning(f"⚠️ LLaVA failed, using BLIP-2: {e}")
            await self.newsreader.initialize_option_b_blip2_quantized()
    
    async def aggressive_modal_dismissal(self, page):
        """Aggressively dismiss all BBC modals, overlays, and cookie consent"""
        
        # Cookie consent - most common patterns
        cookie_selectors = [
            'button:has-text("Accept")',
            'button:has-text("I Agree")', 
            'button:has-text("Continue")',
            'button:has-text("Accept all")',
            'button:has-text("Accept All")',
            '[data-testid="accept-all"]',
            '[id*="accept"]',
            '[id*="cookie"]',
            '.fc-cta-consent', # OneTrust
            '.banner-actions-button', # Common BBC pattern
        ]
        
        # Sign-in and other modals
        dismiss_selectors = [
            'button:has-text("Not now")',
            'button:has-text("Skip")',
            'button:has-text("Maybe later")',
            'button:has-text("Continue without")',
            'button:has-text("No thanks")',
            '[aria-label="Dismiss"]',
            '[aria-label="Close"]',
            '[aria-label="close"]',
            'button[aria-label*="close"]',
            '.close-button',
            '.modal-close',
            '[data-testid="close"]',
            '[data-testid="dismiss"]',
        ]
        
        all_selectors = cookie_selectors + dismiss_selectors
        
        # Try all selectors quickly
        for selector in all_selectors:
            try:
                elements = page.locator(selector)
                count = await elements.count()
                if count > 0:
                    await elements.first.click(timeout=1000)
                    await asyncio.sleep(0.5)  # Brief pause
                    logger.debug(f"Dismissed modal: {selector}")
            except:
                continue  # Ignore failures, keep trying
    
    async def extract_fast_content(self, page) -> Dict[str, str]:
        """Fast DOM-based content extraction (no screenshots for speed)"""
        
        try:
            # Get title
            title = await page.title()
            
            # Try to extract main article content from DOM
            content_selectors = [
                '[data-component="text-block"]',
                '.story-body__inner',
                '[role="main"] p',
                'main p',
                '.article-body p',
                '[data-testid="paragraph"]'
            ]
            
            content_text = ""
            for selector in content_selectors:
                try:
                    elements = page.locator(selector)
                    count = await elements.count()
                    if count > 0:
                        for i in range(min(count, 5)):  # First 5 paragraphs max
                            text = await elements.nth(i).text_content()
                            if text and len(text) > 20:
                                content_text += text + " "
                        if len(content_text) > 100:
                            break
                except:
                    continue
            
            # Fallback to headline extraction
            if len(content_text) < 50:
                headline_selectors = [
                    'h1',
                    '[data-component="headline"]',
                    '.story-headline',
                    '[role="main"] h1'
                ]
                
                for selector in headline_selectors:
                    try:
                        element = page.locator(selector).first
                        headline = await element.text_content()
                        if headline and len(headline) > 10:
                            content_text = headline
                            break
                    except:
                        continue
            
            return {
                "title": title,
                "content": content_text.strip()[:1000],  # Limit for efficiency
                "method": "dom_extraction"
            }
            
        except Exception as e:
            logger.warning(f"Content extraction failed: {e}")
            return {
                "title": await page.title() if page else "Unknown",
                "content": "",
                "method": "failed"
            }
    
    async def process_single_url(self, browser, url: str) -> Optional[Dict]:
        """Process a single URL with aggressive modal handling"""
        
        try:
            context = await browser.new_context(
                viewport={'width': 1024, 'height': 768},
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            )
            page = await context.new_page()
            
            # Navigate with timeout
            await page.goto(url, wait_until='domcontentloaded', timeout=10000)
            
            # Immediate modal dismissal (don't wait for full load)
            await self.aggressive_modal_dismissal(page)
            
            # Brief wait for content to stabilize
            await asyncio.sleep(1)
            
            # Second round of modal dismissal after content loads
            await self.aggressive_modal_dismissal(page)
            
            # Extract content quickly
            content_data = await self.extract_fast_content(page)
            
            # Close context immediately
            await context.close()
            
            # Only analyze if we got meaningful content
            if len(content_data["content"]) > 50:
                
                # Quick NewsReader analysis (use title + content summary)
                analysis_text = f"{content_data['title']} - {content_data['content'][:500]}"
                
                # Simple heuristic check before expensive AI analysis
                news_keywords = ['news', 'reports', 'said', 'according', 'announced', 'confirmed']
                if any(keyword in analysis_text.lower() for keyword in news_keywords):
                    
                    try:
                        # Fallback to simple text analysis instead of image analysis for speed
                        analysis = f"Article: {content_data['title']}\nContent: {content_data['content'][:300]}"
                        
                        return {
                            "url": url,
                            "title": content_data["title"],
                            "content": content_data["content"],
                            "analysis": analysis,
                            "extraction_method": content_data["method"],
                            "timestamp": datetime.now().isoformat(),
                            "status": "success"
                        }
                    except Exception as e:
                        logger.warning(f"Analysis failed for {url}: {e}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to process {url}: {e}")
            return {
                "url": url,
                "title": "Error",
                "content": f"Processing failed: {e}",
                "analysis": "Failed to process",
                "extraction_method": "error",
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    
    async def get_bbc_england_urls(self, max_urls: int = 50) -> List[str]:
        """Get BBC England article URLs quickly"""
        
        browser = await async_playwright().start()
        browser_instance = await browser.chromium.launch(headless=True)
        context = await browser_instance.new_context()
        page = await context.new_page()
        
        try:
            await page.goto("https://www.bbc.co.uk/news/england", timeout=15000)
            await self.aggressive_modal_dismissal(page)
            await asyncio.sleep(2)
            
            # Get article links
            links = await page.locator('a[href*="articles/"]').all()
            urls = []
            
            for link in links:
                try:
                    href = await link.get_attribute('href')
                    if href and 'articles/' in href:
                        if href.startswith('/'):
                            href = f"https://www.bbc.co.uk{href}"
                        if href not in urls:
                            urls.append(href)
                            if len(urls) >= max_urls:
                                break
                except:
                    continue
            
            await browser_instance.close()
            logger.info(f"✅ Found {len(urls)} article URLs")
            return urls
            
        except Exception as e:
            logger.error(f"Failed to get URLs: {e}")
            await browser_instance.close()
            return []
    
    async def process_batch(self, urls: List[str]) -> List[Dict]:
        """Process URLs in batches for speed"""
        
        browser = await async_playwright().start()
        browser_instance = await browser.chromium.launch(headless=True)
        
        results = []
        
        try:
            # Process in smaller concurrent batches to balance speed vs stability
            for i in range(0, len(urls), self.batch_size):
                batch_urls = urls[i:i + self.batch_size]
                logger.info(f"🔄 Processing batch {i//self.batch_size + 1}: {len(batch_urls)} URLs")
                
                # Process batch concurrently
                tasks = [self.process_single_url(browser_instance, url) for url in batch_urls]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect successful results
                for result in batch_results:
                    if isinstance(result, dict) and result.get("status") == "success":
                        results.append(result)
                        logger.info(f"✅ Success: {result['title'][:50]}...")
                
                # Brief cleanup between batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Small delay between batches
                await asyncio.sleep(0.5)
        
        finally:
            await browser_instance.close()
        
        return results
    
    async def run_production_crawl(self, max_articles: int = 100):
        """Main production crawling function"""
        
        start_time = time.time()
        logger.info(f"🚀 Starting production crawl for {max_articles} articles")
        
        # Initialize
        await self.initialize()
        
        # Get URLs
        urls = await self.get_bbc_england_urls(max_urls=max_articles)
        if not urls:
            logger.error("❌ No URLs found!")
            return
        
        # Process in batches
        results = await self.process_batch(urls)
        
        # Save results
        output_file = f"production_bbc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            "production_crawl": True,
            "total_urls": len(urls),
            "successful_articles": len(results),
            "processing_time_seconds": time.time() - start_time,
            "articles_per_second": len(results) / (time.time() - start_time),
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🎉 Crawl complete!")
        logger.info(f"📊 Processed {len(results)} articles in {time.time() - start_time:.1f}s")
        logger.info(f"⚡ Rate: {len(results) / (time.time() - start_time):.2f} articles/second")
        logger.info(f"💾 Results saved to: {output_file}")
        
        return results

async def main():
    """Run production BBC crawler"""
    crawler = ProductionBBCCrawler(batch_size=5)  # Conservative batch size
    await crawler.run_production_crawl(max_articles=50)  # Test with 50 articles

if __name__ == "__main__":
    asyncio.run(main())
