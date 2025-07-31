#!/usr/bin/env python3
"""
Debug Scout Agent Crawling - Check what's actually being extracted
"""

import requests
import json

print("🔍 DEBUG: Scout Agent Crawling Analysis")
print("=" * 50)

# Test with minimal parameters
test_payload = {
    "args": ["https://www.reuters.com"],
    "kwargs": {
        "max_depth": 1,
        "max_pages": 3,
        "word_count_threshold": 10,  # Very low threshold
        "analyze_content": False     # Skip Scout intelligence for now
    }
}

print(f"📋 Test parameters:")
print(f"  URL: {test_payload['args'][0]}")
print(f"  Max depth: {test_payload['kwargs']['max_depth']}")
print(f"  Max pages: {test_payload['kwargs']['max_pages']}")
print(f"  Min words: {test_payload['kwargs']['word_count_threshold']}")
print(f"  Scout analysis: {test_payload['kwargs']['analyze_content']}")

try:
    print("\n📡 Sending request to Scout Agent...")
    response = requests.post("http://localhost:8002/enhanced_deep_crawl_site", 
                           json=test_payload, 
                           timeout=90)
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Response received! Type: {type(data)}")
        print(f"📊 Articles found: {len(data)}")
        
        if data:
            print("\n📄 Article Analysis:")
            for i, article in enumerate(data):
                print(f"\n{i+1}. Article Details:")
                print(f"   Title: {article.get('title', 'NO TITLE')}")
                print(f"   URL: {article.get('url', 'NO URL')}")
                print(f"   Content length: {len(article.get('content', ''))}")
                print(f"   Word count: {article.get('word_count', 0)}")
                print(f"   Depth: {article.get('depth', 'N/A')}")
                print(f"   Source method: {article.get('source_method', 'N/A')}")
                
                # Show content preview
                content = article.get('content', '')
                if content:
                    preview = content[:200].replace('\n', ' ').strip()
                    print(f"   Content preview: {preview}...")
                else:
                    print(f"   Content preview: EMPTY")
        else:
            print("\n❌ No articles found")
            print("🔍 This suggests:")
            print("  1. Content extraction is failing")
            print("  2. Word count threshold too high") 
            print("  3. BestFirstCrawlingStrategy not finding content")
            
        # Show raw response for analysis
        print(f"\n🔧 Raw response: {json.dumps(data, indent=2)[:500]}...")
            
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text[:500]}...")
        
except Exception as e:
    print(f"❌ Exception: {e}")

print("\n" + "="*50)
print("💡 Next steps if no articles found:")
print("  1. Check Scout Agent logs for crawl4ai errors")
print("  2. Test with different URLs")
print("  3. Try fallback crawling method")
print("  4. Check if crawl4ai BestFirstCrawlingStrategy is working")
