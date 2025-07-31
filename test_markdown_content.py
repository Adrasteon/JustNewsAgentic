#!/usr/bin/env python3
"""
Test Crawl4AI Markdown Content - Better Alternative?
"""

import asyncio
from crawl4ai import AsyncWebCrawler

async def test_markdown_content():
    """Test if markdown content is cleaner than cleaned_html"""
    
    print("🔍 CRAWL4AI MARKDOWN CONTENT TEST")
    print("=" * 50)
    
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(
                url="https://www.bbc.com/news",
                timeout=30,
                bypass_cache=True
            )
            
            if result.success and hasattr(result, 'markdown') and result.markdown:
                markdown_content = result.markdown
                
                print(f"📊 MARKDOWN ANALYSIS:")
                print(f"Length: {len(markdown_content)} characters")
                print(f"Word count: {len(markdown_content.split())}")
                
                print(f"\n📄 MARKDOWN PREVIEW (first 800 chars):")
                print("-" * 50)
                print(markdown_content[:800])
                print("-" * 50)
                
                # Check for cleanliness
                html_tags = ['<html>', '<div>', '<nav>', '<header>', '<footer>']
                tags_found = [tag for tag in html_tags if tag in markdown_content]
                
                if tags_found:
                    print(f"❌ MARKDOWN STILL HAS HTML: {tags_found}")
                else:
                    print(f"✅ Markdown appears clean of HTML tags")
                
                # Check for UI elements
                ui_elements = ['menu', 'navigation', 'sidebar', 'cookie']
                ui_found = [elem for elem in ui_elements if elem.lower() in markdown_content.lower()]
                
                print(f"\n🔍 CONTENT QUALITY CHECK:")
                if ui_found:
                    print(f"⚠️  UI elements still present: {ui_found}")
                else:
                    print(f"✅ No obvious UI elements")
                
                # Show structure
                lines = [line.strip() for line in markdown_content.split('\n') if line.strip()]
                print(f"\n📝 FIRST 10 NON-EMPTY LINES:")
                for i, line in enumerate(lines[:10]):
                    print(f"{i+1}. {line[:80]}{'...' if len(line) > 80 else ''}")
                
                return markdown_content
            else:
                print("❌ No markdown content available")
                return None
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    content = asyncio.run(test_markdown_content())
    
    if content:
        print(f"\n💡 RECOMMENDATION:")
        if len([line for line in content.split('\n') if line.strip() and not line.startswith('#')]) > 10:
            print("✅ Markdown content looks usable - much better than 'cleaned' HTML")
            print("   Suggest: Use markdown instead of cleaned_html in Scout Agent")
        else:
            print("❌ Markdown content also problematic")
    
    print("\n" + "="*50)
