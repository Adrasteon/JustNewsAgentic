#!/usr/bin/env python3
"""
Simple Scout Agent Test - Focus on getting articles
"""

import requests
import json

print("🔍 Testing Scout Agent - Simple Crawl")
print("=" * 50)

# Test with a simple news site
test_payload = {
    "args": ["https://www.reuters.com"],
    "kwargs": {
        "max_depth": 1,
        "max_pages": 3,
        "word_count_threshold": 50  # Very low threshold
    }
}

try:
    print("📡 Sending request to Scout Agent...")
    response = requests.post("http://localhost:8002/enhanced_deep_crawl_site", 
                           json=test_payload, 
                           timeout=60)
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Found {len(data)} articles")
        
        if data:
            print("\n📄 Article Details:")
            for i, article in enumerate(data[:2]):  # Show first 2
                print(f"\n{i+1}. Title: {article.get('title', 'No title')[:80]}...")
                print(f"   URL: {article.get('url', 'No URL')}")
                print(f"   Content length: {len(article.get('content', ''))}")
                print(f"   Word count: {article.get('word_count', 0)}")
        else:
            print("❌ No articles found")
            print("Let's check what the response contains:")
            print(f"Response data: {data}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text[:500]}...")
        
except Exception as e:
    print(f"❌ Exception: {e}")

print("\n" + "="*50)
