#!/usr/bin/env python3
"""
Improved BBC Crawler with Cookie Handling

This version properly handles BBC's cookie consent and sign-in requirements
to get to the actual news content.
"""

import asyncio
import json
import sys
import gc
import torch
from datetime import datetime
from playwright.async_api import async_playwright
from PIL import Image
import io
import random

async def get_bbc_article_urls(max_urls=20):
    """Get multiple BBC article URLs"""
    print("🔍 Collecting BBC England article URLs...")
    
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context()
    page = await context.new_page()
    
    try:
        await page.goto("https://www.bbc.co.uk/news/england", wait_until='networkidle')
        await asyncio.sleep(3)
        
        # Get all article links
        links = await page.locator('a[href*="articles/"]').all()
        urls = []
        
        for link in links:
            try:
                href = await link.get_attribute('href')
                if href and 'articles/' in href and 'sport' not in href:
                    if href.startswith('/'):
                        href = f"https://www.bbc.co.uk{href}"
                    if href not in urls:
                        urls.append(href)
                        if len(urls) >= max_urls:
                            break
            except:
                continue
        
        await browser.close()
        print(f"✅ Found {len(urls)} unique article URLs")
        return urls
        
    except Exception as e:
        print(f"❌ Error getting URLs: {e}")
        await browser.close()
        return []

async def process_single_article_with_cookies(url):
    """Process a single BBC article with proper cookie handling"""
    
    try:
        print(f"🔄 Processing: {url}")
        
        # Add crash_safe_newsreader to path
        sys.path.append('/home/adra/JustNewsAgentic')
        from crash_safe_newsreader import CrashSafeNewsReader
        
        # Initialize NewsReader with very conservative settings
        newsreader = CrashSafeNewsReader(max_gpu_memory_gb=10.0)
        
        success = await newsreader.initialize_with_memory_safety()
        if not success:
            print("❌ NewsReader initialization failed")
            return None
        
        # Capture screenshot with cookie handling
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1024, 'height': 768})
        page = await context.new_page()
        
        # Navigate to article
        await page.goto(url, wait_until='networkidle')
        await asyncio.sleep(2)
        
        # Handle cookie consent if present
        try:
            cookie_accept = page.locator('button:has-text("Accept"), button:has-text("I Agree"), button:has-text("Continue")')
            if await cookie_accept.count() > 0:
                print("🍪 Accepting cookies...")
                await cookie_accept.first.click()
                await asyncio.sleep(2)
        except:
            pass
        
        # Try to dismiss any sign-in prompts
        try:
            dismiss_buttons = page.locator('button:has-text("Not now"), button:has-text("Skip"), button:has-text("Maybe later"), button:has-text("Continue without"), [aria-label="Dismiss"], [aria-label="Close"]')
            if await dismiss_buttons.count() > 0:
                print("❌ Dismissing sign-in prompt...")
                await dismiss_buttons.first.click()
                await asyncio.sleep(2)
        except:
            pass
        
        # Wait for content to load
        await asyncio.sleep(3)
        
        title = await page.title()
        
        # Scroll down to get more content visible
        await page.evaluate("window.scrollTo(0, window.innerHeight)")
        await asyncio.sleep(1)
        
        screenshot_bytes = await page.screenshot(full_page=False, type='png')
        
        await browser.close()
        
        # Analyze with NewsReader
        image = Image.open(io.BytesIO(screenshot_bytes))
        
        # Conservative image size
        if max(image.size) > 600:
            image.thumbnail((600, 400), Image.Resampling.LANCZOS)
        
        print("🤖 Analyzing with NewsReader...")
        analysis = await newsreader.safe_analyze_content(
            image, 
            "Extract the main headline, key points, and summary from this BBC news article. Focus on the actual news content, not website navigation or sign-up prompts."
        )
        
        # Immediate cleanup
        newsreader.cleanup()
        del newsreader
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Create result
        result = {
            "url": url,
            "title": title,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        print(f"✅ Article processed!")
        print(f"📰 Title: {title}")
        print(f"🔍 Analysis preview: {analysis[:100]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing {url}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "url": url,
            "title": "Error",
            "analysis": f"Processing failed: {e}",
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }

async def main():
    """Main batch processing function"""
    print("🚀 Improved BBC Article Processor")
    print("==================================")
    
    # Get article URLs
    urls = await get_bbc_article_urls(max_urls=20)
    
    if not urls:
        print("❌ No URLs found!")
        return
    
    print(f"📊 Will process {len(urls)} articles...")
    
    results = []
    
    for i, url in enumerate(urls, 1):
        print(f"\n🔄 [{i}/{len(urls)}] Processing article {i}...")
        
        result = await process_single_article_with_cookies(url)
        if result:
            results.append(result)
        
        # Delay between articles (except for the last one)
        if i < len(urls):
            delay = random.randint(5, 10)  # Random delay to be more natural
            print(f"⏳ Waiting {delay} seconds before next article...")
            await asyncio.sleep(delay)
    
    # Save combined results
    summary = {
        "batch_processing": True,
        "total_processed": len(results),
        "processing_method": "improved_cookie_handling",
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    output_file = f"improved_bbc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Processing complete!")
    print(f"📄 Results saved to: {output_file}")
    print(f"✅ Successfully processed {len([r for r in results if r['status'] == 'success'])} articles")
    print(f"❌ Failed: {len([r for r in results if r['status'] == 'error'])} articles")

if __name__ == "__main__":
    asyncio.run(main())
