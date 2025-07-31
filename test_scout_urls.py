#!/usr/bin/env python3
"""
Test Scout Agent with different URLs
"""

import requests
import json

test_urls = [
    "https://www.reuters.com",
    "https://news.ycombinator.com",
    "https://www.theguardian.com/world"
]

for url in test_urls:
    print(f"\n🔍 Testing URL: {url}")
    try:
        response = requests.post("http://localhost:8002/enhanced_deep_crawl_site", 
                               json={"args": [url], 
                                    "kwargs": {"max_depth": 1, "max_pages": 3}}, 
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data)} articles")
            
            if data:
                # Show first article details
                first_article = data[0]
                print(f"Sample article:")
                print(f"  Title: {first_article.get('title', 'No title')[:60]}...")
                print(f"  URL: {first_article.get('url', 'No URL')}")
                print(f"  Content length: {len(first_article.get('content', ''))}")
                print(f"  Word count: {first_article.get('word_count', 0)}")
                break  # Success, stop testing
            else:
                print("❌ No articles found")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

print("\n" + "="*50)
print("Scout Agent functionality test complete")
