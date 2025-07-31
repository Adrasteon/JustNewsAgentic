#!/usr/bin/env python3
"""
Direct Crawl4AI Test - Bypass Scout Agent complexity
"""

import asyncio
from crawl4ai import AsyncWebCrawler

async def test_direct_crawl():
    print("🔍 Direct Crawl4AI Test (Bypassing Scout Agent)")
    print("=" * 50)
    
    try:
        async with AsyncWebCrawler(verbose=True) as crawler:
            print("📡 Testing simple crawl...")
            
            # Simple crawl first
            result = await crawler.arun(
                url="https://www.reuters.com",
                timeout=30,
                page_timeout=20,
                bypass_cache=True
            )
            
            if result.success:
                print(f"✅ Simple crawl successful!")
                print(f"📄 Title: {result.metadata.get('title', 'No title')}")
                print(f"📊 Content length: {len(result.cleaned_html) if result.cleaned_html else 0}")
                print(f"🔗 Links found: {len(result.links.get('internal', [])) if result.links else 0}")
                
                if result.cleaned_html:
                    word_count = len(result.cleaned_html.split())
                    print(f"📝 Word count: {word_count}")
                    
                    # Show content preview
                    preview = result.cleaned_html[:300].replace('\n', ' ').strip()
                    print(f"📖 Content preview: {preview}...")
                    
                    return True
                else:
                    print("❌ No content extracted")
                    return False
            else:
                print(f"❌ Simple crawl failed: {result.error_message}")
                return False
                
    except Exception as e:
        print(f"❌ Direct crawl4ai error: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_direct_crawl())
    
    if result:
        print("\n✅ Crawl4AI is working! The issue is in Scout Agent's implementation.")
    else:
        print("\n❌ Crawl4AI itself has issues. Need to debug crawl4ai setup.")
        
    print("\n" + "="*50)
