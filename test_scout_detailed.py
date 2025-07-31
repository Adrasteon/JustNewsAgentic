#!/usr/bin/env python3
"""
Test Scout Agent with correct parameter format
"""

import requests
import json

print("Testing Scout Agent with correct parameter format...")

# Test with URL in args (first position)
test_payload = {
    "args": ["https://www.reuters.com"],
    "kwargs": {
        "max_depth": 1,
        "max_pages": 5,
        "word_count_threshold": 100  # Lower threshold for testing
    }
}

print(f"Test payload: {test_payload}")

try:
    response = requests.post("http://localhost:8002/enhanced_deep_crawl_site", 
                           json=test_payload, 
                           timeout=60)
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response type: {type(data)}")
        print(f"Articles found: {len(data)}")
        
        if data:
            # Show first article details
            first_article = data[0]
            print(f"\nFirst article details:")
            print(f"  Title: {first_article.get('title', 'No title')}")
            print(f"  URL: {first_article.get('url', 'No URL')}")
            print(f"  Content length: {len(first_article.get('content', ''))}")
            print(f"  Word count: {first_article.get('word_count', 0)}")
            print(f"  Depth: {first_article.get('depth', 'N/A')}")
        else:
            print("❌ Still no articles found")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Exception: {e}")

print("\n" + "="*50)
