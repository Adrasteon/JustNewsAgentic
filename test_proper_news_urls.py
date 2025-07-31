#!/usr/bin/env python3
"""
Test Scout Agent with proper news URLs
"""

import requests
import json

print("🔍 Testing Scout Agent with Proper News URLs")
print("=" * 50)

# Test with actual news section URLs that should have articles
test_urls = [
    "https://www.reuters.com/world/",           # Reuters World News
    "https://www.bbc.com/news",                 # BBC News
    "https://www.cnn.com/world",                # CNN World
    "https://news.ycombinator.com",             # Hacker News (simple)
]

for url in test_urls:
    print(f"\n🔍 Testing URL: {url}")
    
    test_payload = {
        "args": [url],
        "kwargs": {
            "max_depth": 1,
            "max_pages": 5,
            "word_count_threshold": 100,
            "analyze_content": False
        }
    }
    
    try:
        response = requests.post("http://localhost:8002/enhanced_deep_crawl_site", 
                               json=test_payload, 
                               timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data)} articles")
            
            if data:
                # Show first article
                article = data[0]
                print(f"  📄 Title: {article.get('title', 'No title')[:60]}...")
                print(f"  🔗 URL: {article.get('url', 'No URL')}")
                print(f"  📊 Words: {article.get('word_count', 0)}")
                
                # SUCCESS! Break on first working URL
                print(f"\n🎉 SUCCESS! Found articles on: {url}")
                break
            else:
                print("❌ No articles found")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

print("\n" + "="*50)
