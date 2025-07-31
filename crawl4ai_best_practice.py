#!/usr/bin/env python3
"""
Final Crawl4AI Best Practice Investigation
Based on evidence: markdown is consistently clean, cleaned_html is not
"""

import asyncio
from crawl4ai import AsyncWebCrawler

async def demonstrate_best_practice():
    """Demonstrate the correct way to use Crawl4AI based on our findings"""
    
    print("✅ CRAWL4AI BEST PRACTICE DEMONSTRATION")
    print("=" * 60)
    print("Based on investigation: markdown content is the clean output")
    
    test_url = "https://www.bbc.com/news"
    
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(
                url=test_url,
                word_count_threshold=50,
                bypass_cache=True,
                remove_overlay_elements=True,
                simulate_user=True
            )
            
            if result.success:
                print(f"✅ Crawl successful for: {result.metadata.get('title', 'Unknown')}")
                
                # Compare outputs
                print(f"\n📊 CONTENT COMPARISON:")
                print(f"Raw HTML:     {len(result.html):,} characters")
                print(f"Cleaned HTML: {len(result.cleaned_html):,} characters")  
                print(f"Markdown:     {len(result.markdown):,} characters")
                
                # Analyze cleanliness
                print(f"\n🔍 CLEANLINESS ANALYSIS:")
                
                # Check cleaned_html
                html_tags_in_cleaned = sum(1 for tag in ['<html>', '<div>', '<nav>', '<header>'] 
                                         if tag in result.cleaned_html)
                print(f"HTML tags in 'cleaned_html': {html_tags_in_cleaned}")
                
                # Check markdown  
                html_tags_in_markdown = sum(1 for tag in ['<html>', '<div>', '<nav>', '<header>'] 
                                          if tag in result.markdown)
                print(f"HTML tags in markdown: {html_tags_in_markdown}")
                
                print(f"\n💡 CONCLUSION:")
                if html_tags_in_markdown == 0 and html_tags_in_cleaned > 0:
                    print("✅ MARKDOWN is the clean content output in Crawl4AI 0.7.2")
                    print("❌ 'cleaned_html' still contains HTML structure")
                    print("🎯 BEST PRACTICE: Use result.markdown for clean text content")
                    
                    # Show clean markdown sample
                    print(f"\n📄 CLEAN MARKDOWN SAMPLE:")
                    print("-" * 40)
                    markdown_lines = [line.strip() for line in result.markdown.split('\n') 
                                    if line.strip() and not line.startswith('##')][:10]
                    for i, line in enumerate(markdown_lines[:5], 1):
                        print(f"{i}. {line[:70]}...")
                    
                    return True
                else:
                    print("❓ Unexpected results - need further investigation")
                    return False
            else:
                print(f"❌ Crawl failed: {result.error_message}")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def test_markdown_quality():
    """Test the quality of markdown content for news processing"""
    
    print(f"\n🧪 MARKDOWN CONTENT QUALITY TEST")
    print("=" * 40)
    
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url="https://www.bbc.com/news",
                word_count_threshold=100,
                bypass_cache=True
            )
            
            if result.success and result.markdown:
                markdown_content = result.markdown
                
                # Quality metrics
                word_count = len(markdown_content.split())
                char_count = len(markdown_content)
                lines = [line.strip() for line in markdown_content.split('\n') if line.strip()]
                
                print(f"📊 QUALITY METRICS:")
                print(f"Word count: {word_count:,}")
                print(f"Character count: {char_count:,}")
                print(f"Lines of content: {len(lines):,}")
                print(f"Average words per line: {word_count/len(lines):.1f}")
                
                # Content analysis
                has_headings = any(line.startswith('#') for line in lines)
                has_links = '[' in markdown_content and '](' in markdown_content
                has_navigation = any(word in markdown_content.lower() 
                                   for word in ['menu', 'navigation', 'skip', 'accessibility'])
                
                print(f"\n📄 CONTENT ANALYSIS:")
                print(f"Contains headings: {'✅' if has_headings else '❌'}")
                print(f"Contains links: {'✅' if has_links else '❌'}")
                print(f"Contains navigation: {'⚠️' if has_navigation else '✅'}")
                
                if word_count > 1000 and not has_navigation:
                    print(f"\n✅ MARKDOWN QUALITY: EXCELLENT for news processing")
                    return True
                elif word_count > 500:
                    print(f"\n✅ MARKDOWN QUALITY: GOOD for news processing")
                    return True
                else:
                    print(f"\n⚠️ MARKDOWN QUALITY: Needs improvement")
                    return False
                    
    except Exception as e:
        print(f"❌ Quality test error: {e}")
        return False

if __name__ == "__main__":
    # Demonstrate best practice
    is_markdown_best = asyncio.run(demonstrate_best_practice())
    
    if is_markdown_best:
        # Test quality
        is_quality_good = asyncio.run(test_markdown_quality())
        
        if is_quality_good:
            print(f"\n🎉 FINAL RECOMMENDATION:")
            print("=" * 50)
            print("✅ Use result.markdown in Crawl4AI for clean content")
            print("✅ This is the proper way to get clean text in Crawl4AI 0.7.2")
            print("✅ Update Scout Agent to use markdown instead of cleaned_html")
            print("\n📝 IMPLEMENTATION:")
            print("Change: result.cleaned_html → result.markdown")
            print("This gives clean, structured content without HTML tags")
    
    print("\n" + "="*60)
