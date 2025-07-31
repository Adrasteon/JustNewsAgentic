#!/usr/bin/env python3
"""
Test Crawl4AI With Actual Current News Articles
"""

import asyncio
from crawl4ai import AsyncWebCrawler
import re

async def test_real_news_articles():
    """Test Crawl4AI with confirmed news articles"""
    
    print("🎯 TESTING ACTUAL NEWS ARTICLES")
    print("=" * 50)
    
    # Let's try some known active news sites with predictable article URLs
    test_articles = [
        # Use BBC homepage to find current top stories
        "https://www.bbc.com/",
        # Try AP News which has simpler URL structure  
        "https://apnews.com/",
        # Try The Guardian
        "https://www.theguardian.com/international",
    ]
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        for homepage in test_articles:
            print(f"\n🔍 FINDING ARTICLES FROM: {homepage}")
            print("-" * 40)
            
            try:
                # Get homepage
                result = await crawler.arun(url=homepage, bypass_cache=True)
                
                if result.success and result.html:
                    # Extract article links from HTML
                    article_urls = extract_article_urls(result.html, homepage)
                    
                    print(f"Found {len(article_urls)} potential articles")
                    
                    # Test the first few
                    for i, url in enumerate(article_urls[:2]):
                        await test_article_content(crawler, url, f"Article {i+1}")
                        
            except Exception as e:
                print(f"❌ Error: {e}")

def extract_article_urls(html_content, base_url):
    """Extract article URLs from homepage HTML"""
    urls = []
    
    # Find all href links
    href_pattern = r'href="([^"]*)"'
    matches = re.findall(href_pattern, html_content)
    
    for match in matches:
        # Make absolute URL
        if match.startswith('/'):
            if 'bbc.com' in base_url:
                match = 'https://www.bbc.com' + match
            elif 'apnews.com' in base_url:
                match = 'https://apnews.com' + match
            elif 'theguardian.com' in base_url:
                match = 'https://www.theguardian.com' + match
        
        # Filter for likely article URLs
        is_article = False
        
        if 'bbc.com' in match:
            # BBC articles: /news/[category]-[id] or /news/articles/[id]
            if re.match(r'https://www\.bbc\.com/news/[\w-]+-\d{8}', match):
                is_article = True
        
        elif 'apnews.com' in match:
            # AP News articles: /article/[long-id]
            if '/article/' in match and len(match.split('/')[-1]) > 10:
                is_article = True
                
        elif 'theguardian.com' in match:
            # Guardian articles: /[category]/[year]/[month]/[day]/[title]
            if re.match(r'https://www\.theguardian\.com/[\w-]+/\d{4}/\d{2}/\d{2}/[\w-]+', match):
                is_article = True
        
        if is_article and match not in urls:
            urls.append(match)
    
    return urls[:5]  # Return first 5

async def test_article_content(crawler, url, test_name):
    """Test content extraction from a single article"""
    
    print(f"\n📰 {test_name}: {url}")
    print("   " + "─" * 60)
    
    try:
        result = await crawler.arun(
            url=url,
            bypass_cache=True,
            word_count_threshold=100,
            remove_overlay_elements=True,
            simulate_user=True
        )
        
        if result.success:
            title = result.metadata.get('title', 'No title')[:50]
            print(f"   📋 Title: {title}...")
            
            # Analyze cleaned_html vs markdown
            analyze_content_quality(result.cleaned_html, "Cleaned HTML", "   ")
            analyze_content_quality(result.markdown, "Markdown", "   ")
            
            # Compare which is better
            html_score = score_content_quality(result.cleaned_html)
            markdown_score = score_content_quality(result.markdown)
            
            print(f"   🏆 WINNER: {'Cleaned HTML' if html_score > markdown_score else 'Markdown'}")
            print(f"       HTML Score: {html_score:.2f} | Markdown Score: {markdown_score:.2f}")
            
            # Show sample of winning content
            winner_content = result.cleaned_html if html_score > markdown_score else result.markdown
            show_content_sample(winner_content, "   ")
            
        else:
            print(f"   ❌ Failed: {result.error_message}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

def analyze_content_quality(content, content_type, indent=""):
    """Analyze the quality of extracted content"""
    
    if not content:
        print(f"{indent}📊 {content_type}: No content")
        return
    
    length = len(content)
    word_count = len(content.split())
    
    # Count HTML tags
    html_tags = content.count('<')
    tag_ratio = (html_tags / max(length, 1)) * 100
    
    # Check for news indicators
    news_words = ['said', 'according', 'reported', 'announced', 'stated', 'told', 'explained']
    news_count = sum(1 for word in news_words if word in content.lower())
    
    # Check for navigation/menu content
    nav_indicators = ['menu', 'navigation', 'subscribe', 'sign up', 'cookie', 'privacy policy']
    nav_count = sum(1 for indicator in nav_indicators if indicator.lower() in content.lower())
    
    print(f"{indent}📊 {content_type}:")
    print(f"{indent}   Length: {length:,} chars | Words: {word_count}")
    print(f"{indent}   HTML Tags: {html_tags} ({tag_ratio:.1f}%)")
    print(f"{indent}   News Language: {news_count} indicators")
    print(f"{indent}   Navigation Content: {nav_count} indicators")

def score_content_quality(content):
    """Score content quality (0-1)"""
    
    if not content:
        return 0
    
    length = len(content)
    word_count = len(content.split())
    
    # Basic metrics
    length_score = min(length / 2000, 1)  # Prefer 2000+ chars
    word_score = min(word_count / 300, 1)  # Prefer 300+ words
    
    # HTML ratio (lower is better)
    html_tags = content.count('<')
    tag_ratio = (html_tags / max(length, 1)) * 100
    html_score = max(0, 1 - (tag_ratio / 10))  # Penalize high HTML
    
    # News language indicators
    news_words = ['said', 'according', 'reported', 'announced', 'stated', 'told']
    news_count = sum(1 for word in news_words if word in content.lower())
    news_score = min(news_count / 3, 1)  # Prefer 3+ news indicators
    
    # Navigation penalty
    nav_indicators = ['menu', 'subscribe', 'cookie', 'privacy policy']
    nav_count = sum(1 for indicator in nav_indicators if indicator.lower() in content.lower())
    nav_penalty = min(nav_count * 0.2, 0.5)  # Up to 50% penalty
    
    total_score = (length_score + word_score + html_score + news_score) / 4 - nav_penalty
    return max(0, min(1, total_score))

def show_content_sample(content, indent=""):
    """Show a sample of the content"""
    
    if not content:
        return
    
    # Clean HTML tags for preview
    clean_text = re.sub(r'<[^>]+>', ' ', content)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # Get first few sentences
    sentences = [s.strip() for s in clean_text.split('.') if len(s.strip()) > 20]
    
    print(f"{indent}📖 Content Sample:")
    for i, sentence in enumerate(sentences[:3]):
        print(f"{indent}   {i+1}. {sentence[:70]}...")

async def direct_article_test():
    """Test with known good article URLs for comparison"""
    
    print(f"\n🧪 DIRECT ARTICLE TEST")
    print("=" * 30)
    
    # Try some different approaches with a single known page
    direct_urls = [
        "https://www.bbc.com/news",  # Known to work but is navigation
        "https://apnews.com/hub/business",  # Business section
    ]
    
    async with AsyncWebCrawler() as crawler:
        for url in direct_urls:
            print(f"\n🎯 Testing: {url}")
            
            try:
                result = await crawler.arun(
                    url=url,
                    bypass_cache=True,
                    word_count_threshold=50
                )
                
                if result.success:
                    html_quality = score_content_quality(result.cleaned_html)
                    markdown_quality = score_content_quality(result.markdown)
                    
                    print(f"   HTML Quality: {html_quality:.2f}")
                    print(f"   Markdown Quality: {markdown_quality:.2f}")
                    
                    if html_quality > 0.3 or markdown_quality > 0.3:
                        print(f"   ✅ Reasonable content quality")
                    else:
                        print(f"   ⚠️ Low content quality - likely navigation page")
                        
            except Exception as e:
                print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🎯 GOAL: Test Crawl4AI with actual current news articles")
    print("🔍 METHOD: Extract article URLs from homepages and test content quality")
    
    asyncio.run(test_real_news_articles())
    asyncio.run(direct_article_test())
    
    print(f"\n💡 KEY INSIGHTS:")
    print("1. Compare cleaned_html vs markdown quality on real articles")
    print("2. Identify navigation content vs article content patterns")
    print("3. Determine best extraction method for Scout Agent")
