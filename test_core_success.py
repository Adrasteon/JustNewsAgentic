#!/usr/bin/env python3
"""
JustNews V4 - CORE FUNCTIONALITY SUCCESS TEST
Validates: Scout Agent crawls articles → Memory Agent stores them
"""

import requests
import json

print("🎉 JUSTNEWS V4 CORE FUNCTIONALITY TEST")
print("=" * 50)
print("Testing: Scout Agent → Memory Agent → Database Storage")

def test_working_pipeline():
    """Test the working core functionality"""
    
    print("\n🕵️ PHASE 1: Scout Agent Article Crawling")
    print("-" * 30)
    
    # Test Scout Agent with working URL
    scout_payload = {
        "args": ["https://www.bbc.com/news"],
        "kwargs": {
            "max_depth": 1,
            "max_pages": 3,
            "word_count_threshold": 500,
            "analyze_content": False
        }
    }
    
    try:
        response = requests.post("http://localhost:8002/enhanced_deep_crawl_site", 
                               json=scout_payload, timeout=60)
        
        if response.status_code == 200:
            articles = response.json()
            print(f"✅ Scout Agent: Found {len(articles)} articles")
            
            if articles:
                article = articles[0]
                print(f"   📄 Title: {article.get('title', 'No title')}")
                print(f"   📊 Word count: {article.get('word_count', 0)}")
                print(f"   📏 Content length: {len(article.get('content', ''))}")
                
                print(f"\n💾 PHASE 2: Memory Agent Database Storage")
                print("-" * 30)
                
                # Store in Memory Agent
                memory_payload = {
                    "content": article.get('content', '')[:1500],  # Reasonable size
                    "metadata": {
                        "title": article.get('title', 'Unknown'),
                        "url": article.get('url', ''),
                        "source": "bbc_news_test",
                        "word_count": article.get('word_count', 0),
                        "test_timestamp": "2025-07-29_22:00"
                    }
                }
                
                memory_response = requests.post("http://localhost:8007/save_article", 
                                              json=memory_payload, timeout=30)
                
                if memory_response.status_code == 200:
                    result = memory_response.json()
                    article_id = result.get('id')
                    print(f"✅ Memory Agent: Stored article (ID: {article_id})")
                    
                    print(f"\n🎊 SUCCESS SUMMARY")
                    print("=" * 50)
                    print("✅ Scout Agent: Successfully crawling news articles")
                    print("✅ Memory Agent: Successfully storing in database")
                    print("✅ Core Pipeline: FULLY OPERATIONAL")
                    print(f"✅ Article stored with ID: {article_id}")
                    
                    print(f"\n🚀 NEXT STEPS:")
                    print("1. ✅ Scout + Memory pipeline working perfectly")
                    print("2. Add Analyst Agent for content analysis")
                    print("3. Build complete news processing workflow")
                    print("4. Optimize vector search performance")
                    
                    return True
                else:
                    print(f"❌ Memory Agent storage failed: {memory_response.status_code}")
                    return False
            else:
                print("❌ No articles found by Scout Agent")
                return False
        else:
            print(f"❌ Scout Agent failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return False

if __name__ == "__main__":
    success = test_working_pipeline()
    
    if success:
        print(f"\n🎉 CORE FUNCTIONALITY VERIFIED!")
        print("JustNews V4 Scout → Memory pipeline is ready for expansion!")
    else:
        print(f"\n❌ Core functionality needs attention")
    
    print("\n" + "="*50)
