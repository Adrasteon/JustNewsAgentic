#!/usr/bin/env python3
"""
Deep Analysis of Article Content Extraction
"""

import asyncio
from crawl4ai import AsyncWebCrawler
import re

async def deep_content_analysis():
    """Deep analysis of the article content to see what we're actually getting"""
    
    article_url = "https://www.bbc.co.uk/news/articles/cy85737235go"
    
    print("🔬 DEEP CONTENT ANALYSIS")
    print("=" * 40)
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(
            url=article_url,
            bypass_cache=True,
            word_count_threshold=50
        )
        
        if result.success:
            print("✅ Article loaded successfully!")
            print(f"Title: {result.metadata.get('title', 'No title')}")
            print()
            
            # Show actual markdown content structure
            if result.markdown:
                print("📖 MARKDOWN CONTENT STRUCTURE:")
                print("-" * 35)
                
                # Split into lines and analyze
                lines = result.markdown.split('\n')
                content_started = False
                article_content = []
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Look for start of actual article content
                    if not content_started:
                        # Skip navigation/header content
                        if any(skip in line.lower() for skip in ['bbc homepage', 'skip to content', 'accessibility', 'menu', 'search']):
                            continue
                        # Look for article headline or story start
                        if ('terror' in line.lower() and 'skyscraper' in line.lower()) or \
                           ('employees' in line.lower() and 'park avenue' in line.lower()) or \
                           line.startswith('For hundreds of'):
                            content_started = True
                    
                    if content_started:
                        article_content.append(line)
                        if len(article_content) >= 10:  # Get first 10 lines of actual content
                            break
                
                print("📄 ACTUAL ARTICLE CONTENT (first 10 lines):")
                for i, line in enumerate(article_content):
                    print(f"   {i+1:2d}. {line[:80]}...")
                
                print(f"\nContent Analysis:")
                print(f"   Total lines: {len(lines)}")
                print(f"   Article content lines: {len(article_content)}")
                print(f"   Navigation lines skipped: {len(lines) - len(article_content)}")
            
            # Also check cleaned_html for comparison
            print(f"\n🔍 CLEANED HTML CONTENT ANALYSIS:")
            print("-" * 40)
            
            if result.cleaned_html:
                # Extract just the text from HTML
                clean_text = re.sub(r'<[^>]+>', '', result.cleaned_html)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                
                # Find the main article content
                paragraphs = [p.strip() for p in clean_text.split('.') if len(p.strip()) > 50]
                
                # Look for paragraphs that seem like article content
                article_paragraphs = []
                for para in paragraphs:
                    if any(indicator in para.lower() for indicator in ['employees', 'workers', 'commuters', 'skyscraper', 'manhattan']):
                        article_paragraphs.append(para)
                        if len(article_paragraphs) >= 5:
                            break
                
                print("📄 ARTICLE PARAGRAPHS FROM HTML:")
                for i, para in enumerate(article_paragraphs):
                    print(f"   {i+1}. {para[:100]}...")
                
                print(f"\nHTML Analysis:")
                print(f"   Total text length: {len(clean_text):,} chars")
                print(f"   Article paragraphs found: {len(article_paragraphs)}")
            
            # Recommendation based on analysis
            print(f"\n🎯 FINAL RECOMMENDATION:")
            print("-" * 25)
            
            if result.markdown and result.cleaned_html:
                # Check which has better article content ratio
                md_lines = len(result.markdown.split('\n'))
                md_content_ratio = len(article_content) / max(md_lines, 1)
                
                html_clean = re.sub(r'<[^>]+>', '', result.cleaned_html)
                html_total_words = len(html_clean.split())
                article_words = sum(len(para.split()) for para in article_paragraphs)
                html_content_ratio = article_words / max(html_total_words, 1)
                
                print(f"Markdown content ratio: {md_content_ratio:.2f}")
                print(f"HTML content ratio: {html_content_ratio:.2f}")
                
                if md_content_ratio > html_content_ratio:
                    print("✅ FINAL CHOICE: Use result.markdown")
                    print("   - Better content-to-navigation ratio")
                    print("   - Cleaner text format")
                else:
                    print("✅ FINAL CHOICE: Use result.cleaned_html with text extraction")
                    print("   - Better article content extraction")
                    print("   - More complete paragraphs")
                
                print(f"\n💡 IMPLEMENTATION:")
                print("Update Scout Agent to use winning method and clean navigation content")

if __name__ == "__main__":
    asyncio.run(deep_content_analysis())
