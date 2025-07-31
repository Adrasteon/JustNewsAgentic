#!/usr/bin/env python3
"""
Test Crawl4AI with Actual News Article URLs
The issue might be that we're testing homepages instead of articles
"""

import asyncio
from crawl4ai import AsyncWebCrawler

async def test_article_urls():
    """Test Crawl4AI with actual news article URLs instead of homepages"""
    
    print("🔍 CRAWL4AI ARTICLE URL TESTING")
    print("=" * 50)
    
    # Test with actual article URLs, not homepages
    article_urls = [
        "https://www.bbc.com/news/world-middle-east-66665800",  # Specific BBC article
        "https://www.reuters.com/world/middle-east/",           # Reuters Middle East section
        "https://edition.cnn.com/world",                       # CNN World section
    ]
    
    for i, url in enumerate(article_urls, 1):
        print(f"\n{i}️⃣ TESTING: {url}")
        print("-" * 50)
        
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=50,
                    bypass_cache=True,
                    remove_overlay_elements=True,
                    simulate_user=True
                )
                
                if result.success:
                    print(f"✅ Crawl successful")
                    print(f"   Title: {result.metadata.get('title', 'No title')}")
                    print(f"   Raw HTML: {len(result.html) if result.html else 0} chars")
                    print(f"   Cleaned HTML: {len(result.cleaned_html) if result.cleaned_html else 0} chars")
                    print(f"   Markdown: {len(result.markdown) if result.markdown else 0} chars")
                    
                    if result.cleaned_html:
                        # Check if cleaned_html is actually clean for articles
                        preview = result.cleaned_html[:400].replace('\n', ' ')
                        print(f"   Cleaned preview: {preview}...")
                        
                        # Check for HTML tags
                        html_tags = ['<html>', '<body>', '<div class=', '<nav>', '<header>']
                        found_tags = [tag for tag in html_tags if tag in result.cleaned_html]
                        
                        if found_tags:
                            print(f"   ❌ Still has HTML: {found_tags[:3]}...")
                        else:
                            print(f"   ✅ Clean text content!")
                            
                            # If clean, show more details
                            lines = [line.strip() for line in result.cleaned_html.split('\n') if line.strip()]
                            print(f"   📄 First few lines of clean content:")
                            for j, line in enumerate(lines[:5]):
                                print(f"      {j+1}. {line[:60]}...")
                            
                            return result.cleaned_html  # Return clean content if found
                    
                    # Also check markdown
                    if result.markdown:
                        markdown_preview = result.markdown[:300].replace('\n', ' ')
                        print(f"   Markdown preview: {markdown_preview}...")
                        
                        # Check if markdown is cleaner
                        if '<html>' not in result.markdown and '<div' not in result.markdown:
                            print(f"   ✅ Markdown is clean!")
                
                else:
                    print(f"❌ Crawl failed: {result.error_message}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return None

async def test_specific_bbc_article():
    """Test with a very specific BBC article URL"""
    
    print(f"\n🎯 TESTING SPECIFIC BBC ARTICLE")
    print("=" * 50)
    
    # Try to find a real BBC article URL
    try:
        # First, let's see what links are available on BBC News
        async with AsyncWebCrawler() as crawler:
            homepage_result = await crawler.arun(
                url="https://www.bbc.com/news",
                bypass_cache=True
            )
            
            if homepage_result.success and homepage_result.links:
                # Look for article links
                article_links = []
                for link_type, links in homepage_result.links.items():
                    for link in links[:5]:  # Check first 5 links
                        if '/news/' in link and 'bbc.com' in link and len(link.split('/')) > 5:
                            article_links.append(link)
                
                print(f"Found potential article links: {len(article_links)}")
                
                if article_links:
                    test_article = article_links[0]
                    print(f"Testing article: {test_article}")
                    
                    article_result = await crawler.arun(
                        url=test_article,
                        bypass_cache=True,
                        word_count_threshold=10
                    )
                    
                    if article_result.success:
                        print(f"✅ Article crawl successful")
                        print(f"   Title: {article_result.metadata.get('title', 'No title')}")
                        print(f"   Cleaned HTML: {len(article_result.cleaned_html)} chars")
                        
                        if article_result.cleaned_html:
                            preview = article_result.cleaned_html[:300]
                            print(f"   Content preview: {preview}...")
                            
                            # Check if this is cleaner
                            if '<html>' not in article_result.cleaned_html:
                                print(f"   ✅ Article content is clean!")
                                return True
                            else:
                                print(f"   ❌ Article content still has HTML structure")
                    
    except Exception as e:
        print(f"❌ Error testing specific article: {e}")
    
    return False

if __name__ == "__main__":
    print("Testing hypothesis: Crawl4AI works better with article URLs than homepages")
    
    # Test with article URLs
    clean_content = asyncio.run(test_article_urls())
    
    if not clean_content:
        # Try to find a real article
        success = asyncio.run(test_specific_bbc_article())
        
        if not success:
            print(f"\n💡 CONCLUSION:")
            print("Even with article URLs, cleaned_html contains HTML structure.")
            print("This suggests we need to:")
            print("1. Check Crawl4AI version and documentation")
            print("2. Look for content extraction parameters we're missing")
            print("3. Use a different extraction method")
    else:
        print(f"\n✅ FOUND CLEAN CONTENT!")
        print("The issue was using homepage URLs instead of article URLs!")
    
    print("\n" + "="*50)
