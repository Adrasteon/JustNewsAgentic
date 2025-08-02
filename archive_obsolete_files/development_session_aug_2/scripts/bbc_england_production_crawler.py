#!/usr/bin/env python3
"""
Production BBC England News Crawler with Crash-Safe NewsReader

This script combines the crash-safe NewsReader with BBC England crawling
to extract actual news article titles and content from 19 BBC articles.

Features:
- Crash-safe memory management
- BBC England article discovery
- Screenshot-based content capture
- LLaVA-powered news content analysis
- Structured output with titles and summaries
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import time
from datetime import datetime

# Production imports
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
logger = logging.getLogger("bbc_england_production")

class BBCEnglandProductionCrawler:
    """
    Production BBC England crawler with crash-safe NewsReader integration.
    
    Designed to extract meaningful news content including titles and summaries
    from BBC England articles using vision-language analysis.
    """
    
    def __init__(self):
        self.newsreader = None
        self.browser = None
        self.context = None
        self.results = []
        self.failed_urls = []
        self.processed_count = 0
        
    async def initialize(self) -> bool:
        """Initialize production crawler with crash-safe NewsReader"""
        try:
            logger.info("🚀 Initializing Production BBC England Crawler...")
            
            # Initialize crash-safe NewsReader
            self.newsreader = CrashSafeNewsReader(max_gpu_memory_gb=18.0)
            newsreader_success = await self.newsreader.initialize_with_memory_safety()
            
            if not newsreader_success:
                logger.error("❌ Failed to initialize crash-safe NewsReader")
                return False
            
            # Check NewsReader health
            health = await self.newsreader.health_check()
            logger.info(f"📊 NewsReader Health: Model={health['model_type']}, GPU={health['gpu_memory_allocated_gb']}GB")
            
            # Initialize browser for screenshot capture
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(headless=True)
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            logger.info("✅ Production crawler initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Crawler initialization failed: {e}")
            return False
    
    async def discover_bbc_england_articles(self, target_count: int = 19) -> List[str]:
        """
        Discover BBC England article URLs from the main England page.
        
        Args:
            target_count: Number of article URLs to find
            
        Returns:
            List[str]: BBC England article URLs
        """
        try:
            logger.info(f"🔍 Discovering {target_count} BBC England article URLs...")
            
            page = await self.context.new_page()
            
            # Navigate to BBC England news page
            await page.goto("https://www.bbc.co.uk/news/england", wait_until='networkidle')
            await asyncio.sleep(3)  # Allow dynamic content to load
            
            urls = set()
            
            # Look for article links using multiple strategies
            selectors = [
                'a[href*="articles/"]',  # New BBC article format
                'a[href*="/news/uk-england-"]',  # Traditional England articles
                'a[href*="/news/england-"]',  # England region articles
                '[data-testid] a[href*="/news/"]',  # Data testid articles
                '.gs-c-promo-heading a[href*="/news/"]',  # Promo headings
            ]
            
            for selector in selectors:
                try:
                    links = await page.locator(selector).all()
                    logger.info(f"📝 Selector '{selector[:30]}...' found {len(links)} links")
                    
                    for link in links:
                        try:
                            href = await link.get_attribute('href')
                            if href:
                                # Convert relative URLs to absolute
                                if href.startswith('/'):
                                    href = f"https://www.bbc.co.uk{href}"
                                
                                # Filter for actual news articles
                                if self._is_valid_news_article(href):
                                    urls.add(href)
                                    
                                # Stop when we have enough URLs
                                if len(urls) >= target_count:
                                    break
                                    
                        except Exception as e:
                            continue  # Skip individual link errors
                            
                    if len(urls) >= target_count:
                        break
                        
                except Exception as e:
                    logger.warning(f"⚠️ Selector failed: {e}")
                    continue
            
            # Convert to list and limit to target count
            article_urls = list(urls)[:target_count]
            
            await page.close()
            
            logger.info(f"✅ Found {len(article_urls)} BBC England article URLs")
            return article_urls
            
        except Exception as e:
            logger.error(f"❌ Error discovering BBC England URLs: {e}")
            return []
    
    def _is_valid_news_article(self, url: str) -> bool:
        """Check if URL is a valid news article"""
        # Include articles and traditional news formats
        valid_patterns = ['articles/', '/news/uk-england-', '/news/england-']
        
        # Exclude non-article content
        exclude_patterns = ['video', 'live', 'sport', 'topics', 'regions', 'weather', 'travel']
        
        has_valid_pattern = any(pattern in url for pattern in valid_patterns)
        has_exclude_pattern = any(pattern in url for pattern in exclude_patterns)
        
        return has_valid_pattern and not has_exclude_pattern
    
    async def capture_article_screenshot(self, url: str) -> Optional[Image.Image]:
        """
        Capture full-page screenshot of BBC article.
        
        Args:
            url: Article URL to capture
            
        Returns:
            Optional[Image.Image]: PIL Image or None if failed
        """
        try:
            page = await self.context.new_page()
            
            # Navigate to article
            await page.goto(url, wait_until='networkidle', timeout=30000)
            await asyncio.sleep(3)  # Wait for dynamic content
            
            # Handle cookie banners
            try:
                cookie_selectors = [
                    'button:has-text("Accept")',
                    'button:has-text("OK")',
                    'button:has-text("Agree")',
                    '[data-testid="banner-accept"]',
                    '.fc-cta-consent'
                ]
                
                for selector in cookie_selectors:
                    cookie_button = page.locator(selector)
                    if await cookie_button.count() > 0:
                        await cookie_button.first.click()
                        await asyncio.sleep(1)
                        break
            except:
                pass  # Cookie handling is optional
            
            # Capture full-page screenshot
            screenshot_bytes = await page.screenshot(
                full_page=True,
                type='png'
            )
            
            await page.close()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(screenshot_bytes))
            
            logger.info(f"✅ Screenshot captured: {url}")
            return image
            
        except Exception as e:
            logger.error(f"❌ Screenshot capture failed for {url}: {e}")
            return None
    
    async def analyze_news_article(self, url: str, image: Image.Image) -> Optional[Dict]:
        """
        Analyze BBC article using crash-safe NewsReader with focus on extracting
        actual news content including headlines and summaries.
        
        Args:
            url: Article URL
            image: Screenshot image to analyze
            
        Returns:
            Optional[Dict]: Analysis result with title and content
        """
        try:
            # Create detailed analysis prompt for news extraction
            context = f"""
            This is a screenshot of a BBC news article from {url}. 
            Please analyze this image and extract:
            1. The main headline/title of the news article
            2. A brief summary of the key story points
            3. Any important details or quotes mentioned
            4. The general topic/category (politics, sports, local news, etc.)
            
            Focus on the actual news content, not website navigation elements.
            """
            
            # Analyze with crash-safe NewsReader
            analysis = await self.newsreader.safe_analyze_content(image, context)
            
            if analysis:
                result = {
                    "url": url,
                    "raw_analysis": analysis,
                    "parsed_content": self._parse_news_content(analysis),
                    "timestamp": datetime.now().isoformat(),
                    "model_type": self.newsreader.model_type,
                    "status": "success"
                }
                
                logger.info(f"✅ Analysis complete: {url}")
                return result
            else:
                logger.warning(f"⚠️ No analysis result for: {url}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Analysis failed for {url}: {e}")
            return None
    
    def _parse_news_content(self, analysis: str) -> Dict:
        """
        Parse the LLaVA analysis to extract structured news content.
        
        Args:
            analysis: Raw analysis text from LLaVA
            
        Returns:
            Dict: Structured news content
        """
        try:
            # Simple parsing to extract key information
            lines = analysis.split('\n')
            
            # Look for title/headline indicators
            title = ""
            summary = ""
            category = ""
            
            # Extract title (usually the first substantial line)
            for line in lines:
                line = line.strip()
                if len(line) > 20 and not line.startswith(('The image', 'This is', 'The screenshot')):
                    title = line
                    break
            
            # Extract summary (remaining content)
            summary_lines = []
            for line in lines:
                line = line.strip()
                if (len(line) > 10 and 
                    not line.startswith(('The image', 'This is', 'The screenshot')) and
                    line != title):
                    summary_lines.append(line)
            
            summary = ' '.join(summary_lines[:3])  # First 3 meaningful lines
            
            # Simple category detection
            analysis_lower = analysis.lower()
            if any(word in analysis_lower for word in ['politics', 'minister', 'government', 'parliament']):
                category = "Politics"
            elif any(word in analysis_lower for word in ['sport', 'football', 'cricket', 'tennis']):
                category = "Sport"
            elif any(word in analysis_lower for word in ['weather', 'rain', 'temperature', 'storm']):
                category = "Weather"
            elif any(word in analysis_lower for word in ['crime', 'police', 'court', 'arrest']):
                category = "Crime"
            else:
                category = "General News"
            
            return {
                "extracted_title": title[:200] if title else "Title not clearly identified",
                "extracted_summary": summary[:500] if summary else "Summary not available",
                "estimated_category": category,
                "analysis_length": len(analysis)
            }
            
        except Exception as e:
            logger.error(f"❌ Content parsing error: {e}")
            return {
                "extracted_title": "Parsing failed",
                "extracted_summary": "Could not parse content",
                "estimated_category": "Unknown",
                "analysis_length": len(analysis) if analysis else 0
            }
    
    async def process_single_article(self, url: str, article_num: int, total: int) -> Optional[Dict]:
        """
        Process a single BBC article with full pipeline.
        
        Args:
            url: Article URL to process
            article_num: Current article number
            total: Total articles being processed
            
        Returns:
            Optional[Dict]: Processing result
        """
        try:
            logger.info(f"📰 [{article_num}/{total}] Processing: {url}")
            
            # Capture screenshot
            image = await self.capture_article_screenshot(url)
            if not image:
                self.failed_urls.append({"url": url, "reason": "screenshot_failed"})
                return None
            
            # Analyze with NewsReader
            result = await self.analyze_news_article(url, image)
            if result:
                self.processed_count += 1
                
                # Log extracted content for monitoring
                parsed = result.get('parsed_content', {})
                title = parsed.get('extracted_title', 'No title')[:80]
                category = parsed.get('estimated_category', 'Unknown')
                
                logger.info(f"📄 Extracted: [{category}] {title}...")
                
                return result
            else:
                self.failed_urls.append({"url": url, "reason": "analysis_failed"})
                return None
                
        except Exception as e:
            logger.error(f"❌ Article processing failed for {url}: {e}")
            self.failed_urls.append({"url": url, "reason": str(e)})
            return None
    
    async def crawl_bbc_england_news(self, target_articles: int = 19) -> Dict:
        """
        Main crawling method to process BBC England articles.
        
        Args:
            target_articles: Number of articles to process
            
        Returns:
            Dict: Complete crawling results with extracted news content
        """
        try:
            start_time = time.time()
            
            logger.info(f"🚀 Starting BBC England news crawling ({target_articles} articles)")
            
            # Discover article URLs
            urls = await self.discover_bbc_england_articles(target_articles)
            if not urls:
                return {"error": "No URLs found", "results": []}
            
            logger.info(f"📝 Processing {len(urls)} BBC England articles...")
            
            # Process articles with crash-safe pipeline
            for i, url in enumerate(urls, 1):
                result = await self.process_single_article(url, i, len(urls))
                if result:
                    self.results.append(result)
                
                # Brief pause between requests to be respectful
                await asyncio.sleep(2)
                
                # Monitor memory health periodically
                if i % 5 == 0:
                    health = await self.newsreader.health_check()
                    memory_gb = health.get('gpu_memory_allocated_gb', 0)
                    logger.info(f"💾 Memory check: {memory_gb}GB GPU usage")
            
            # Compile final results
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            
            # Get final system health
            final_health = await self.newsreader.health_check()
            
            crawl_results = {
                "success": True,
                "total_urls_discovered": len(urls),
                "successful_analyses": len(self.results),
                "failed_analyses": len(self.failed_urls),
                "success_rate": round((len(self.results) / len(urls) * 100), 1) if urls else 0,
                "processing_time_seconds": duration,
                "avg_time_per_article": round(duration / len(urls), 1) if urls else 0,
                "newsreader_model": self.newsreader.model_type,
                "final_system_health": final_health,
                "extracted_articles": self.results,
                "failed_urls": self.failed_urls,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ BBC England crawling complete: {len(self.results)}/{len(urls)} successful")
            return crawl_results
            
        except Exception as e:
            logger.error(f"❌ BBC England crawling failed: {e}")
            return {"error": str(e), "results": self.results}
    
    async def cleanup(self):
        """Production cleanup with proper resource management"""
        try:
            if self.context:
                await self.context.close()
            
            if self.browser:
                await self.browser.close()
            
            if self.newsreader:
                self.newsreader.cleanup()
            
            logger.info("✅ Production crawler cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

async def main():
    """Main execution function for BBC England news crawler"""
    
    crawler = BBCEnglandProductionCrawler()
    
    try:
        # Initialize production crawler
        if not await crawler.initialize():
            logger.error("❌ Failed to initialize production crawler")
            return
        
        # Run BBC England crawling for 19 articles
        results = await crawler.crawl_bbc_england_news(target_articles=19)
        
        # Save detailed results
        output_file = "bbc_england_news_extraction.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Display comprehensive summary
        print("\n" + "="*90)
        print("🏆 BBC ENGLAND NEWS EXTRACTION RESULTS")
        print("="*90)
        print(f"📊 Success Rate: {results.get('success_rate', 0)}%")
        print(f"✅ Successful: {results.get('successful_analyses', 0)}")
        print(f"❌ Failed: {results.get('failed_analyses', 0)}")
        print(f"⏱️  Total Duration: {results.get('processing_time_seconds', 0)}s")
        print(f"📈 Avg per Article: {results.get('avg_time_per_article', 0)}s")
        print(f"🤖 Model: {results.get('newsreader_model', 'unknown')}")
        
        # Display extracted content samples
        print(f"\n📰 EXTRACTED NEWS CONTENT SAMPLES:")
        print("-" * 90)
        
        for i, article in enumerate(results.get('extracted_articles', [])[:5], 1):
            parsed = article.get('parsed_content', {})
            title = parsed.get('extracted_title', 'No title')
            summary = parsed.get('extracted_summary', 'No summary')
            category = parsed.get('estimated_category', 'Unknown')
            url = article.get('url', '')
            
            print(f"{i}. [{category}] {title}")
            print(f"   Summary: {summary[:150]}{'...' if len(summary) > 150 else ''}")
            print(f"   URL: {url}")
            print()
        
        print(f"📄 Complete results saved: {output_file}")
        print("="*90)
        
    finally:
        # Always cleanup
        await crawler.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
