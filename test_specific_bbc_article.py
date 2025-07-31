#!/usr/bin/env python3
"""
Test Specific BBC Article with Crawl4AI
"""

import asyncio
from crawl4ai import AsyncWebCrawler
import re

async def test_specific_article():
    """Test the specific BBC article provided by user"""
    
    article_url = "https://www.bbc.co.uk/news/articles/cy85737235go"
    
    print("🎯 TESTING SPECIFIC BBC ARTICLE")
    print("=" * 50)
    print(f"URL: {article_url}")
    print()
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        try:
            result = await crawler.arun(
                url=article_url,
                bypass_cache=True,
                word_count_threshold=50,
                remove_overlay_elements=True,
                simulate_user=True
            )
            
            if result.success:
                print("✅ Article loaded successfully!")
                print()
                
                # Get basic info
                title = result.metadata.get('title', 'No title')
                print(f"📋 Title: {title}")
                print()
                
                # Analyze cleaned_html
                print("🔍 CLEANED HTML ANALYSIS:")
                print("-" * 30)
                if result.cleaned_html:
                    html_len = len(result.cleaned_html)
                    html_words = len(result.cleaned_html.split())
                    html_tags = result.cleaned_html.count('<')
                    html_tag_ratio = (html_tags / max(html_len, 1)) * 100
                    
                    print(f"Length: {html_len:,} characters")
                    print(f"Words: {html_words:,}")
                    print(f"HTML Tags: {html_tags} ({html_tag_ratio:.1f}%)")
                    
                    # Check for article content indicators
                    news_indicators = ['said', 'according to', 'reported', 'announced', 'told', 'stated']
                    found_indicators = [word for word in news_indicators if word in result.cleaned_html.lower()]
                    print(f"News Language: {len(found_indicators)} indicators found: {found_indicators}")
                    
                    # Check for navigation/menu content
                    nav_indicators = ['menu', 'navigation', 'subscribe', 'sign up', 'cookie policy', 'privacy']
                    found_nav = [word for word in nav_indicators if word.lower() in result.cleaned_html.lower()]
                    print(f"Navigation Content: {len(found_nav)} indicators: {found_nav}")
                    
                    # Show first few sentences
                    clean_text = re.sub(r'<[^>]+>', '', result.cleaned_html)
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                    sentences = [s.strip() for s in clean_text.split('.') if len(s.strip()) > 20]
                    
                    print(f"\n📖 First 3 sentences (cleaned HTML):")
                    for i, sentence in enumerate(sentences[:3]):
                        print(f"   {i+1}. {sentence}")
                
                print()
                
                # Analyze markdown
                print("🔍 MARKDOWN ANALYSIS:")
                print("-" * 25)
                if result.markdown:
                    md_len = len(result.markdown)
                    md_words = len(result.markdown.split())
                    has_html = '<' in result.markdown
                    
                    print(f"Length: {md_len:,} characters")
                    print(f"Words: {md_words:,}")
                    print(f"Contains HTML: {'Yes' if has_html else 'No'}")
                    
                    # Show first few sentences
                    sentences = [s.strip() for s in result.markdown.split('.') if len(s.strip()) > 20]
                    
                    print(f"\n📖 First 3 sentences (markdown):")
                    for i, sentence in enumerate(sentences[:3]):
                        print(f"   {i+1}. {sentence}")
                
                print()
                
                # Comparison and recommendation
                print("🏆 COMPARISON & RECOMMENDATION:")
                print("-" * 35)
                
                if result.cleaned_html and result.markdown:
                    html_quality = assess_quality(result.cleaned_html, "HTML")
                    md_quality = assess_quality(result.markdown, "Markdown")
                    
                    if html_quality > md_quality:
                        print("✅ RECOMMENDATION: Use cleaned_html")
                        print(f"   HTML Quality Score: {html_quality:.2f}")
                        print(f"   Markdown Quality Score: {md_quality:.2f}")
                    else:
                        print("✅ RECOMMENDATION: Use markdown")
                        print(f"   Markdown Quality Score: {md_quality:.2f}")
                        print(f"   HTML Quality Score: {html_quality:.2f}")
                    
                    # Show the winner content sample
                    winner = result.cleaned_html if html_quality > md_quality else result.markdown
                    winner_type = "cleaned_html" if html_quality > md_quality else "markdown"
                    
                    print(f"\n📄 WINNING CONTENT SAMPLE ({winner_type}):")
                    clean_sample = re.sub(r'<[^>]+>', '', winner) if '<' in winner else winner
                    clean_sample = re.sub(r'\s+', ' ', clean_sample).strip()
                    print(f"   {clean_sample[:200]}...")
                
            else:
                print(f"❌ Failed to load article: {result.error_message}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def assess_quality(content, content_type):
    """Assess content quality with scoring"""
    
    if not content:
        return 0
    
    length = len(content)
    words = len(content.split())
    
    # Length score (prefer 1000+ chars)
    length_score = min(length / 1000, 1)
    
    # Word count score (prefer 150+ words)
    word_score = min(words / 150, 1)
    
    # HTML cleanliness (lower HTML tags is better)
    html_tags = content.count('<')
    tag_ratio = (html_tags / max(length, 1)) * 100
    clean_score = max(0, 1 - (tag_ratio / 5))  # Penalize >5% HTML tags
    
    # News content indicators
    news_words = ['said', 'according', 'reported', 'announced', 'told', 'stated']
    news_count = sum(1 for word in news_words if word in content.lower())
    news_score = min(news_count / 3, 1)  # Prefer 3+ news indicators
    
    # Navigation content penalty
    nav_words = ['menu', 'subscribe', 'cookie', 'privacy', 'sign up']
    nav_count = sum(1 for word in nav_words if word.lower() in content.lower())
    nav_penalty = min(nav_count * 0.1, 0.3)  # Up to 30% penalty
    
    total_score = (length_score + word_score + clean_score + news_score) / 4 - nav_penalty
    final_score = max(0, min(1, total_score))
    
    print(f"   {content_type} Quality Breakdown:")
    print(f"     Length: {length_score:.2f} | Words: {word_score:.2f}")
    print(f"     Cleanliness: {clean_score:.2f} | News Content: {news_score:.2f}")
    print(f"     Navigation Penalty: -{nav_penalty:.2f}")
    print(f"     Final Score: {final_score:.2f}")
    print()
    
    return final_score

if __name__ == "__main__":
    print("🎯 Testing specific BBC article to determine best extraction method")
    print("🔍 This will help us configure the Scout Agent properly")
    
    asyncio.run(test_specific_article())
    
    print("\n💡 NEXT STEPS:")
    print("1. Update Scout Agent tools.py with winning extraction method")
    print("2. Test Scout → Memory pipeline with real article content")
    print("3. Validate that articles are properly stored in database")
