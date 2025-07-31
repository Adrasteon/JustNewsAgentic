#!/usr/bin/env python3
"""
Investigate Proper Crawl4AI Usage for Content Extraction
"""

import asyncio
from crawl4ai import AsyncWebCrawler

async def test_proper_crawl4ai_usage():
    """Test different Crawl4AI configurations to find the correct approach"""
    
    print("🔍 CRAWL4AI PROPER USAGE INVESTIGATION")
    print("=" * 60)
    
    test_url = "https://www.bbc.com/news"
    
    # Test 1: Basic crawl with proper content extraction parameters
    print("\n1️⃣ TEST 1: Basic crawl with content extraction settings")
    print("-" * 40)
    
    try:
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(
                url=test_url,
                # Content extraction parameters
                word_count_threshold=10,
                extraction_strategy="NoExtractionStrategy",  # Let's see what this does
                chunking_strategy="RegexChunking",
                bypass_cache=True
            )
            
            if result.success:
                print(f"✅ Basic crawl successful")
                print(f"   Raw HTML: {len(result.html) if result.html else 0} chars")
                print(f"   Cleaned HTML: {len(result.cleaned_html) if result.cleaned_html else 0} chars")
                print(f"   Markdown: {len(result.markdown) if result.markdown else 0} chars")
                
                if result.cleaned_html:
                    preview = result.cleaned_html[:200].replace('\n', ' ')
                    print(f"   Cleaned preview: {preview}...")
                    
                    # Check if it's actually clean
                    if '<html>' in result.cleaned_html or '<div>' in result.cleaned_html:
                        print(f"   ❌ Still contains HTML structure")
                    else:
                        print(f"   ✅ Appears to be clean text")
            else:
                print(f"❌ Basic crawl failed: {result.error_message}")
                
    except Exception as e:
        print(f"❌ Test 1 error: {e}")
    
    # Test 2: Try with different extraction strategy
    print("\n2️⃣ TEST 2: With LLMExtractionStrategy (if available)")
    print("-" * 40)
    
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=test_url,
                word_count_threshold=10,
                bypass_cache=True,
                # Try to get just the main content
                css_selector="main, article, .content, #content",  # Common content selectors
                exclude_tags=['nav', 'header', 'footer', 'aside', 'menu']
            )
            
            if result.success:
                print(f"✅ CSS selector crawl successful")
                print(f"   Cleaned HTML: {len(result.cleaned_html) if result.cleaned_html else 0} chars")
                
                if result.cleaned_html:
                    preview = result.cleaned_html[:300].replace('\n', ' ')
                    print(f"   Content preview: {preview}...")
                    
                    # Check cleanliness
                    nav_elements = ['<nav', '<header', '<footer', 'menu', 'navigation']
                    found_nav = [elem for elem in nav_elements if elem in result.cleaned_html]
                    
                    if found_nav:
                        print(f"   ⚠️ Still has navigation: {found_nav}")
                    else:
                        print(f"   ✅ Navigation elements removed")
            else:
                print(f"❌ CSS selector crawl failed: {result.error_message}")
                
    except Exception as e:
        print(f"❌ Test 2 error: {e}")
    
    # Test 3: Try with content-only extraction
    print("\n3️⃣ TEST 3: Content-only extraction approach")
    print("-" * 40)
    
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=test_url,
                word_count_threshold=10,
                bypass_cache=True,
                # Focus on content extraction
                remove_overlay_elements=True,
                simulate_user=True,
                wait_for_images=False,
                # Try to exclude common non-content elements
                exclude_domains=[],
                exclude_tags=['script', 'style', 'nav', 'header', 'footer', 'aside']
            )
            
            if result.success:
                print(f"✅ Content-focused crawl successful")
                
                # Try different content access methods
                if hasattr(result, 'fit_markdown') and result.fit_markdown:
                    print(f"   Fit markdown: {len(result.fit_markdown)} chars")
                    preview = result.fit_markdown[:300]
                    print(f"   Fit markdown preview: {preview}...")
                
                if hasattr(result, 'extracted_content') and result.extracted_content:
                    print(f"   Extracted content: {len(result.extracted_content)} chars")
                
                # Check available attributes
                print(f"   Available result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                
            else:
                print(f"❌ Content-focused crawl failed")
                
    except Exception as e:
        print(f"❌ Test 3 error: {e}")

if __name__ == "__main__":
    asyncio.run(test_proper_crawl4ai_usage())
    
    print(f"\n💡 NEXT STEPS:")
    print("1. Identify which approach gives clean content")
    print("2. Update Scout Agent with proper Crawl4AI configuration")
    print("3. Test with real news article URLs (not homepage)")
    print("\n" + "="*60)
