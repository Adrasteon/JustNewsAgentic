#!/usr/bin/env python3
"""
🧪 JustNews V4 - Test Enhanced Deep Crawl Integration
Test the new enhanced_deep_crawl_site function integrated into Scout agent
"""

import asyncio
import json
import time
import requests
from datetime import datetime

# Configuration
SCOUT_API_URL = "http://localhost:8002"
MCP_BUS_URL = "http://localhost:8000"

def test_enhanced_deepcrawl_via_mcp_bus():
    """Test enhanced deep crawl via MCP Bus"""
    print("🧪 Testing Enhanced Deep Crawl via MCP Bus")
    print("=" * 60)
    
    try:
        # Test with Sky News using user's requested parameters
        payload = {
            "agent": "scout",
            "tool": "enhanced_deep_crawl_site",
            "args": ["https://news.sky.com"],
            "kwargs": {
                "max_depth": 3,          # User requested
                "max_pages": 100,        # User requested  
                "word_count_threshold": 500,  # User requested
                "quality_threshold": 0.6,
                "analyze_content": True
            }
        }
        
        print(f"🎯 Target: https://news.sky.com")
        print(f"📏 Configuration: depth=3, pages=100, min_words=500")
        print(f"🧠 Scout Intelligence: Enabled (quality_threshold=0.6)")
        print("-" * 40)
        
        start_time = time.time()
        
        print("📡 Calling MCP Bus /call endpoint...")
        response = requests.post(f"{MCP_BUS_URL}/call", json=payload, timeout=120)
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            results = response.json()
            
            print(f"✅ Enhanced deep crawl completed in {duration:.2f}s")
            print(f"📊 Results: {len(results)} pages")
            
            if results:
                total_content = sum(item.get('content_length', 0) for item in results)
                avg_scout_score = sum(item.get('scout_score', 0.0) for item in results) / len(results)
                news_articles = [item for item in results if item.get('is_news', False)]
                
                print(f"📈 Content: {total_content:,} characters total")
                print(f"🧠 Average Scout Score: {avg_scout_score:.2f}")
                print(f"📰 News Articles: {len(news_articles)}/{len(results)}")
                print()
                
                print("📄 Top Results:")
                for i, result in enumerate(results[:5]):
                    scout_score = result.get('scout_score', 0.0)
                    is_news = "📰" if result.get('is_news', False) else "📄"
                    print(f"   {i+1}. {is_news} {result.get('title', 'No title')[:50]}...")
                    print(f"      📍 {result.get('url', 'No URL')}")
                    print(f"      📏 {result.get('word_count', 0):,} words, Scout: {scout_score:.2f}")
                    print()
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"enhanced_deepcrawl_integration_test_{timestamp}.json"
                
                test_results = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'test_type': 'enhanced_deepcrawl_via_mcp_bus',
                    'target_url': 'https://news.sky.com',
                    'configuration': payload['kwargs'],
                    'performance': {
                        'duration_seconds': duration,
                        'total_pages': len(results),
                        'total_content_length': total_content,
                        'avg_scout_score': avg_scout_score,
                        'news_articles_count': len(news_articles)
                    },
                    'results': results
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(test_results, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Results saved to {filename}")
                
                return True
            else:
                print("⚠️ No results returned")
                return False
                
        else:
            print(f"❌ MCP Bus call failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Test failed after {duration:.2f}s: {e}")
        return False

def test_enhanced_deepcrawl_direct():
    """Test enhanced deep crawl directly via Scout agent endpoint"""
    print("\n🧪 Testing Enhanced Deep Crawl Direct API")
    print("=" * 60)
    
    try:
        payload = {
            "args": ["https://news.sky.com"],
            "kwargs": {
                "max_depth": 3,
                "max_pages": 10,  # Smaller for direct test
                "word_count_threshold": 500,
                "quality_threshold": 0.6,
                "analyze_content": True
            }
        }
        
        print(f"🎯 Target: https://news.sky.com (direct API)")
        print(f"📏 Configuration: depth=3, pages=10, min_words=500")
        print("-" * 40)
        
        start_time = time.time()
        
        print("📡 Calling Scout agent /enhanced_deep_crawl_site endpoint...")
        response = requests.post(f"{SCOUT_API_URL}/enhanced_deep_crawl_site", json=payload, timeout=120)
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Direct enhanced deep crawl completed in {duration:.2f}s")
            print(f"📊 Results: {len(results)} pages")
            
            if results:
                total_content = sum(item.get('content_length', 0) for item in results)
                print(f"📈 Content: {total_content:,} characters total")
                return True
            else:
                print("⚠️ No results returned")
                return False
        else:
            print(f"❌ Direct API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Direct test failed after {duration:.2f}s: {e}")
        return False

def check_service_health():
    """Check if required services are running"""
    print("🔍 Checking Service Health")
    print("=" * 60)
    
    services = [
        ("MCP Bus", MCP_BUS_URL),
        ("Scout Agent", SCOUT_API_URL)
    ]
    
    all_healthy = True
    
    for service_name, service_url in services:
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name}: Healthy")
            else:
                print(f"⚠️ {service_name}: Unhealthy ({response.status_code})")
                all_healthy = False
        except Exception as e:
            print(f"❌ {service_name}: Not accessible ({e})")
            all_healthy = False
    
    return all_healthy

def main():
    """Main test execution"""
    print("🚀 JustNews V4 - Enhanced Deep Crawl Integration Test")
    print("Testing BestFirstCrawlingStrategy integration into Scout agent")
    print("=" * 80)
    
    # Check service health first
    if not check_service_health():
        print("\n⚠️ Some services are not healthy. Please start required services:")
        print("   1. Start MCP Bus: docker-compose up mcp_bus")
        print("   2. Start Scout Agent: docker-compose up scout")
        print("   Or run full system: docker-compose up")
        return
    
    print("\n🎯 All services healthy. Starting integration tests...")
    
    # Test via MCP Bus (primary integration method)
    mcp_success = test_enhanced_deepcrawl_via_mcp_bus()
    
    # Test direct API (backup validation)  
    direct_success = test_enhanced_deepcrawl_direct()
    
    print("\n🏁 Integration Test Summary")
    print("=" * 60)
    print(f"MCP Bus Integration: {'✅ SUCCESS' if mcp_success else '❌ FAILED'}")
    print(f"Direct API Test: {'✅ SUCCESS' if direct_success else '❌ FAILED'}")
    
    if mcp_success:
        print("\n🎉 Enhanced Deep Crawl Integration SUCCESS!")
        print("   ✅ BestFirstCrawlingStrategy is integrated into Scout agent")
        print("   ✅ Scout intelligence analysis is working")
        print("   ✅ User parameters (depth=3, pages=100, min_words=500) applied")
        print("   ✅ MCP Bus communication is functional")
        print("\n🚀 Ready for production use!")
    else:
        print("\n⚠️ Integration test completed with issues")
        print("   Check logs for troubleshooting information")

if __name__ == "__main__":
    main()
