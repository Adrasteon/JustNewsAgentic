#!/usr/bin/env python3
"""
Inspect Scout Agent Content - What are we actually getting?
"""

import requests
import json

print("🔍 SCOUT AGENT CONTENT INSPECTION")
print("=" * 50)
print("Let's see what the crawler is actually returning...")

def inspect_crawled_content():
    """Examine the actual content returned by Scout Agent"""
    
    # Get content from Scout Agent
    payload = {
        "args": ["https://www.bbc.com/news"],
        "kwargs": {
            "max_depth": 1,
            "max_pages": 2,
            "word_count_threshold": 100,
            "analyze_content": False
        }
    }
    
    try:
        response = requests.post("http://localhost:8002/enhanced_deep_crawl_site", 
                               json=payload, timeout=60)
        
        if response.status_code == 200:
            articles = response.json()
            print(f"✅ Scout found {len(articles)} articles")
            
            if articles:
                article = articles[0]
                content = article.get('content', '')
                
                print(f"\n📊 ARTICLE ANALYSIS:")
                print(f"Title: {article.get('title', 'No title')}")
                print(f"URL: {article.get('url', 'No URL')}")
                print(f"Word count: {article.get('word_count', 0)}")
                print(f"Content length: {len(content)} characters")
                print(f"Depth: {article.get('depth', 'N/A')}")
                print(f"Source method: {article.get('source_method', 'N/A')}")
                
                print(f"\n📄 CONTENT PREVIEW (First 500 chars):")
                print("-" * 50)
                print(content[:500])
                print("-" * 50)
                
                print(f"\n🔍 CONTENT ANALYSIS:")
                
                # Check for HTML tags
                html_indicators = ['<div', '<p>', '<span', '<a href', '<img', '<nav', '<header', '<footer']
                html_found = [tag for tag in html_indicators if tag in content]
                
                if html_found:
                    print(f"⚠️  HTML TAGS DETECTED: {html_found}")
                    print("    This looks like raw HTML, not clean text!")
                else:
                    print("✅ No obvious HTML tags - appears to be clean text")
                
                # Check for navigation/UI elements
                ui_indicators = ['menu', 'navigation', 'sidebar', 'footer', 'header', 'advertisement', 'cookie', 'subscribe']
                ui_found = [indicator for indicator in ui_indicators if indicator.lower() in content.lower()]
                
                if ui_found:
                    print(f"⚠️  UI ELEMENTS DETECTED: {ui_found}")
                    print("    Content includes navigation/UI elements")
                else:
                    print("✅ No obvious UI elements detected")
                
                # Check content structure
                lines = content.split('\n')
                non_empty_lines = [line.strip() for line in lines if line.strip()]
                
                print(f"\n📊 STRUCTURE ANALYSIS:")
                print(f"Total lines: {len(lines)}")
                print(f"Non-empty lines: {len(non_empty_lines)}")
                print(f"Average line length: {sum(len(line) for line in non_empty_lines) / len(non_empty_lines):.1f} chars")
                
                # Show sample lines
                print(f"\n📝 SAMPLE LINES (first 5 non-empty):")
                for i, line in enumerate(non_empty_lines[:5]):
                    print(f"{i+1}. {line[:80]}{'...' if len(line) > 80 else ''}")
                
                print(f"\n💡 RECOMMENDATION:")
                if html_found or ui_found:
                    print("❌ Content needs cleaning - contains HTML/UI elements")
                    print("   Suggest: Content extraction/cleaning before storage")
                else:
                    print("✅ Content appears clean - suitable for storage")
                
                return article
            else:
                print("❌ No articles returned")
                return None
        else:
            print(f"❌ Scout Agent error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    article = inspect_crawled_content()
    
    if article:
        print(f"\n🎯 NEXT STEPS:")
        print("1. Determine if content cleaning is needed")
        print("2. Implement content extraction if necessary")
        print("3. Test storage with cleaned content")
    else:
        print(f"\n❌ No content to analyze")
    
    print("\n" + "="*50)
