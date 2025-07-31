#!/usr/bin/env python3
"""
Proper Crawl4AI Investigation - Test with Actual Article Pages
"""

import asyncio
from crawl4ai import AsyncWebCrawler
import re

async def find_and_test_real_articles():
    """Find actual article URLs and test Crawl4AI properly"""
    
    print("🔍 PROPER CRAWL4AI INVESTIGATION")
    print("=" * 60)
    print("Testing with ACTUAL ARTICLE PAGES, not navigation pages")
    
    # Step 1: Get actual article links from BBC News
    print("\n1️⃣ FINDING REAL ARTICLE URLS")
    print("-" * 30)
    
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            # Crawl the headlines page to find article links
            headlines_result = await crawler.arun(
                url="https://www.bbc.com/news",
                bypass_cache=True
            )
            
            if headlines_result.success:
                # Extract article URLs from the links
                article_urls = []
                
                if headlines_result.links and 'internal' in headlines_result.links:
                    for link in headlines_result.links['internal']:
                        # Look for actual article URLs (they have specific patterns)
                        if (re.match(r'https://www\.bbc\.com/news/[\w-]+-\d+', link) or
                            re.match(r'https://www\.bbc\.com/news/world-[\w-]+-\d+', link)):
                            article_urls.append(link)
                
                print(f"Found {len(article_urls)} potential article URLs")
                
                # Test with the first few actual articles
                for i, article_url in enumerate(article_urls[:3]):
                    await test_actual_article(crawler, article_url, i+1)
                    
            else:
                print("❌ Failed to get headlines page")
                
    except Exception as e:
        print(f"❌ Error finding articles: {e}")
    
    # Step 2: Test with known article URLs
    print(f"\n2️⃣ TESTING WITH KNOWN ARTICLE URLS")
    print("-" * 30)
    
    known_article_urls = [
        "https://www.reuters.com/world/europe/putin-warns-west-russia-could-use-nuclear-weapons-2024-09-25/",
        "https://www.theguardian.com/world/2024/sep/25/ukraine-war-latest-news",
        # These might be outdated, but show the pattern
    ]
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        for i, url in enumerate(known_article_urls):
            print(f"\n🔗 Testing: {url}")
            try:
                result = await crawler.arun(
                    url=url,
                    bypass_cache=True,
                    word_count_threshold=50
                )
                
                if result.success:
                    await analyze_content_quality(result, f"Article {i+1}")
                else:
                    print(f"❌ Failed to crawl: {result.error_message}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")

async def test_actual_article(crawler, article_url, test_num):
    """Test Crawl4AI with an actual article URL"""
    
    print(f"\n📰 TEST {test_num}: {article_url}")
    print("-" * 50)
    
    try:
        result = await crawler.arun(
            url=article_url,
            bypass_cache=True,
            word_count_threshold=100
        )
        
        if result.success:
            await analyze_content_quality(result, f"Article {test_num}")
        else:
            print(f"❌ Article crawl failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Error testing article: {e}")

async def analyze_content_quality(result, test_name):
    """Analyze the quality of extracted content"""
    
    print(f"✅ {test_name} crawl successful")
    print(f"   Title: {result.metadata.get('title', 'No title')[:60]}...")
    
    # Compare different content formats
    raw_html_len = len(result.html) if result.html else 0
    cleaned_html_len = len(result.cleaned_html) if result.cleaned_html else 0
    markdown_len = len(result.markdown) if result.markdown else 0
    
    print(f"   📊 Content lengths:")
    print(f"      Raw HTML: {raw_html_len:,} chars")
    print(f"      Cleaned HTML: {cleaned_html_len:,} chars")
    print(f"      Markdown: {markdown_len:,} chars")
    
    # Analyze cleaned_html quality
    if result.cleaned_html:
        print(f"\n   🔍 CLEANED_HTML ANALYSIS:")
        
        # Check for HTML structure
        html_tags = ['<html>', '<head>', '<body>', '<div>', '<nav>', '<header>', '<footer>']
        found_tags = [tag for tag in html_tags if tag in result.cleaned_html]
        
        if found_tags:
            print(f"      ❌ Contains HTML structure: {found_tags[:3]}...")
        else:
            print(f"      ✅ No HTML structure detected")
        
        # Check for navigation elements
        nav_indicators = ['menu', 'navigation', 'skip to', 'cookie', 'subscribe', 'sign in']
        found_nav = [ind for ind in nav_indicators if ind.lower() in result.cleaned_html.lower()]
        
        if found_nav:
            print(f"      ⚠️ Contains navigation: {found_nav[:3]}...")
        else:
            print(f"      ✅ No navigation elements")
        
        # Show content preview
        clean_preview = result.cleaned_html[:300].replace('\n', ' ').strip()
        print(f"      📄 Preview: {clean_preview}...")
        
        # Determine if this looks like article content
        word_count = len(result.cleaned_html.split())
        sentences = result.cleaned_html.count('.') + result.cleaned_html.count('!') + result.cleaned_html.count('?')
        
        print(f"      📊 Content metrics:")
        print(f"         Words: {word_count}")
        print(f"         Sentences: {sentences}")
        print(f"         Avg words/sentence: {word_count/max(sentences,1):.1f}")
        
        if word_count > 200 and sentences > 10 and not found_tags and not found_nav:
            print(f"      ✅ HIGH QUALITY: Looks like clean article content!")
            return True
        elif word_count > 100 and not found_tags:
            print(f"      ✅ GOOD QUALITY: Reasonably clean content")
            return True
        else:
            print(f"      ❌ LOW QUALITY: Not suitable for article storage")
            return False
    
    # Also analyze markdown
    if result.markdown:
        print(f"\n   🔍 MARKDOWN ANALYSIS:")
        markdown_preview = result.markdown[:200].replace('\n', ' ')
        print(f"      📄 Preview: {markdown_preview}...")
        
        # Check if markdown is better
        nav_in_markdown = sum(1 for ind in ['menu', 'skip to', 'cookie'] 
                            if ind.lower() in result.markdown.lower())
        
        if nav_in_markdown == 0:
            print(f"      ✅ Markdown appears clean")
        else:
            print(f"      ⚠️ Markdown contains navigation elements")
    
    return False

if __name__ == "__main__":
    print("🎯 HYPOTHESIS: Crawl4AI works properly with actual article URLs")
    print("Previous tests used navigation/headlines pages, not articles")
    
    asyncio.run(find_and_test_real_articles())
    
    print(f"\n💡 NEXT STEPS:")
    print("1. If cleaned_html works well with articles → use cleaned_html")
    print("2. If both fail → investigate extraction strategies")
    print("3. Update Scout Agent with proper article URL handling")
    
    print("\n" + "="*60)
