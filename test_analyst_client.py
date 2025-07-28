#!/usr/bin/env python3
"""
Test Client for JustNews V4 Native GPU Analyst
Send real articles to the running analyst service and measure performance
"""

import requests
import time
import json

def test_analyst_service():
    """Test the running JustNews V4 analyst service"""
    
    base_url = "http://localhost:8004"
    
    print("=" * 70)
    print("🧪 Testing JustNews V4 Native GPU Analyst Service")
    print("=" * 70)
    
    # Test health endpoint
    print("🔍 Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Service is running!")
            print(f"   GPU: {health_data.get('gpu', 'Unknown')}")
            print(f"   Platform: {health_data.get('platform', 'Unknown')}")
            print(f"   Expected Performance: {health_data.get('performance', {})}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to analyst service!")
        print("   Make sure 'python start_native_gpu_analyst.py' is running")
        return
    
    # Test built-in benchmark
    print(f"\n⚡ Built-in Performance Benchmark...")
    try:
        response = requests.get(f"{base_url}/performance/benchmark")
        if response.status_code == 200:
            benchmark_data = response.json()
            
            sentiment = benchmark_data.get('sentiment_analysis', {})
            bias = benchmark_data.get('bias_analysis', {})
            
            print(f"✅ Benchmark Results:")
            print(f"   Articles tested: {benchmark_data.get('benchmark_articles', 0)}")
            print(f"   Sentiment Analysis: {sentiment.get('articles_per_second', 0):.1f} articles/sec")
            print(f"   Bias Analysis: {bias.get('articles_per_second', 0):.1f} articles/sec")
            print(f"   GPU: {benchmark_data.get('gpu_device', 'Unknown')}")
        else:
            print(f"❌ Benchmark failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Benchmark error: {e}")
    
    # Test with realistic articles
    print(f"\n📰 Testing with Realistic News Articles...")
    
    realistic_articles = [
        "Breaking news from financial markets today as major technology stocks experienced significant volatility amid concerns about regulatory changes and market uncertainty. Industry analysts are closely monitoring the situation as investors react to new policies that could impact future growth prospects. The developments come at a critical time when market sentiment has been particularly sensitive to geopolitical tensions and economic indicators from major economies around the world.",
        
        "Climate scientists have announced breakthrough findings that could reshape our understanding of environmental patterns and weather systems across multiple regions. The comprehensive study, conducted over several years, involved data collection from numerous monitoring stations and advanced modeling techniques. Researchers found that certain atmospheric phenomena are occurring more frequently than previously predicted, with significant implications for agriculture and urban planning.",
        
        "Technology innovation continues to accelerate as companies announce breakthrough developments in artificial intelligence, quantum computing, and biotechnology sectors. These advances promise to transform industries ranging from healthcare and finance to manufacturing and entertainment. The pace of innovation has been particularly notable in areas where interdisciplinary approaches are yielding unexpected results and opening new possibilities.",
        
        "Economic indicators show mixed results this quarter as various sectors respond differently to policy changes and market conditions. Manufacturing data suggests steady growth while service sectors face ongoing challenges related to workforce availability and supply chain disruptions. Analysts are closely watching consumer spending patterns and employment figures for signs of broader economic trends.",
        
        "Environmental policy initiatives receive varying levels of public support as communities weigh economic impacts against long-term sustainability goals. Recent surveys indicate growing awareness of climate issues among younger demographics, while concerns about implementation costs remain significant factors in policy discussions. Local governments are exploring innovative approaches to balance environmental and economic priorities."
    ]
    
    print(f"   Articles to analyze: {len(realistic_articles)}")
    print(f"   Average article length: {sum(len(a) for a in realistic_articles) // len(realistic_articles)} characters")
    
    # Test sentiment analysis
    print(f"\n🎯 Testing Sentiment Analysis...")
    try:
        payload = {"articles": realistic_articles}
        
        start_time = time.time()
        response = requests.post(f"{base_url}/analyze/sentiment/batch", json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            result_data = response.json()
            total_time = end_time - start_time
            articles_per_sec = len(realistic_articles) / total_time
            
            print(f"✅ Sentiment Analysis Results:")
            print(f"   Articles processed: {result_data.get('count', 0)}")
            print(f"   Processing time: {total_time:.3f}s")
            print(f"   Performance: {articles_per_sec:.1f} articles/sec")
            print(f"   Average per article: {total_time / len(realistic_articles) * 1000:.1f}ms")
            
            # Show sample results
            results = result_data.get('results', [])
            if results:
                print(f"   Sample results:")
                for i, result in enumerate(results[:3]):  # Show first 3
                    print(f"     Article {i+1}: {result}")
        else:
            print(f"❌ Sentiment analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Sentiment test error: {e}")
    
    # Test bias analysis
    print(f"\n⚖️  Testing Bias Analysis...")
    try:
        payload = {"articles": realistic_articles}
        
        start_time = time.time()
        response = requests.post(f"{base_url}/analyze/bias/batch", json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            result_data = response.json()
            total_time = end_time - start_time
            articles_per_sec = len(realistic_articles) / total_time
            
            print(f"✅ Bias Analysis Results:")
            print(f"   Articles processed: {result_data.get('count', 0)}")
            print(f"   Processing time: {total_time:.3f}s")
            print(f"   Performance: {articles_per_sec:.1f} articles/sec")
            print(f"   Average per article: {total_time / len(realistic_articles) * 1000:.1f}ms")
            
            # Show sample results
            results = result_data.get('results', [])
            if results:
                print(f"   Sample results:")
                for i, result in enumerate(results[:3]):  # Show first 3
                    print(f"     Article {i+1}: {result}")
        else:
            print(f"❌ Bias analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Bias test error: {e}")
    
    print(f"\n" + "=" * 70)
    print(f"🏆 JustNews V4 Analyst Service Test Complete!")
    print(f"=" * 70)

if __name__ == "__main__":
    test_analyst_service()
