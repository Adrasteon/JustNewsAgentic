#!/usr/bin/env python3
"""
Direct Crawl4AI Test - Check if cleaned_html works
"""

import asyncio
from crawl4ai import AsyncWebCrawler

async def test_crawl4ai_cleaning():
    """Test crawl4ai's content cleaning directly"""
    
    print("🔍 DIRECT CRAWL4AI CONTENT CLEANING TEST")
    print("=" * 50)
    
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            print("📡 Testing crawl4ai content cleaning...")
            
            result = await crawler.arun(
                url="https://www.bbc.com/news",
                timeout=30,
                page_timeout=20,
                bypass_cache=True,
                remove_overlay_elements=True,
                simulate_user=True
            )
            
            if result.success:
                print(f"✅ Crawl successful!")
                
                # Check different content types
                raw_html = result.html
                cleaned_html = result.cleaned_html
                markdown = getattr(result, 'markdown', None)
                
                print(f"\n📊 CONTENT COMPARISON:")
                print(f"Raw HTML length: {len(raw_html) if raw_html else 0}")
                print(f"Cleaned HTML length: {len(cleaned_html) if cleaned_html else 0}")
                print(f"Markdown length: {len(markdown) if markdown else 0}")
                
                if cleaned_html:
                    print(f"\n📄 CLEANED HTML PREVIEW (first 500 chars):")
                    print("-" * 50)
                    print(cleaned_html[:500])
                    print("-" * 50)
                    
                    # Check if it's actually clean
                    html_tags = ['<html>', '<head>', '<body>', '<nav>', '<header>', '<footer>']
                    tags_found = [tag for tag in html_tags if tag in cleaned_html]
                    
                    if tags_found:
                        print(f"❌ CLEANED HTML STILL HAS TAGS: {tags_found}")
                        print("   Crawl4AI cleaning is not working properly!")
                    else:
                        print(f"✅ Cleaned HTML appears to be actually clean")
                        
                    # Show structure
                    lines = cleaned_html.split('\n')[:10]
                    print(f"\n📝 FIRST 10 LINES OF CLEANED CONTENT:")
                    for i, line in enumerate(lines):
                        if line.strip():
                            print(f"{i+1}. {line.strip()[:80]}...")
                
                else:
                    print("❌ No cleaned_html returned!")
                    
                return result
            else:
                print(f"❌ Crawl failed: {result.error_message}")
                return None
                
    except Exception as e:
        print(f"❌ Direct crawl4ai error: {e}")
        return None

if __name__ == "__main__":
    result = asyncio.run(test_crawl4ai_cleaning())
    
    if result:
        print(f"\n💡 ANALYSIS:")
        if result.cleaned_html and '<html>' not in result.cleaned_html:
            print("✅ Crawl4AI cleaning works - issue is elsewhere")
        else:
            print("❌ Crawl4AI cleaning is not working - this is the root issue")
            print("   Need to fix crawl4ai configuration or use alternative cleaning")
    
    print("\n" + "="*50)
