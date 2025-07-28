#!/usr/bin/env python3
"""
JustNews V4 Conservative Performance Test
Gentle testing with realistic news articles - won't crash your system!
"""

import torch
import time
import gc
from transformers import pipeline
import numpy as np

def create_sample_news_articles(count=10):
    """Generate a small sample of realistic news articles"""
    
    # One realistic news article template (~1,200 characters)
    template = """Breaking news from the financial markets today as major technology stocks experienced significant volatility amid concerns about regulatory changes and market uncertainty. Industry analysts are closely monitoring the situation as investors react to new policies that could impact future growth prospects. The developments come at a critical time when market sentiment has been particularly sensitive to geopolitical tensions and economic indicators. Market participants are awaiting further clarification from regulatory bodies about the scope and timeline of these potential changes. Early trading sessions showed mixed results across different sectors, with some defensive stocks gaining ground while growth-oriented investments faced selling pressure. This pattern reflects the broader uncertainty that has characterized recent market behavior. Experts suggest that investors should maintain a diversified approach and focus on long-term fundamentals rather than short-term market fluctuations."""
    
    articles = []
    for i in range(count):
        # Add slight variation to each article
        variation = f" Update {i+1}: Additional context from recent developments in the sector continues to influence market dynamics."
        articles.append(template + variation)
    
    return articles

def safe_gpu_test():
    """Conservative GPU availability and memory test"""
    
    print("🔍 GPU Status Check:")
    
    if not torch.cuda.is_available():
        print("   ⚠️ CUDA not available - using CPU")
        return False, 0
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"   ✅ GPU: {gpu_name}")
    print(f"   💾 Total Memory: {gpu_memory:.1f} GB")
    
    # Check available memory
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0) / 1e9
    cached = torch.cuda.memory_reserved(0) / 1e9
    
    print(f"   📊 Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    
    return True, gpu_memory

def conservative_sentiment_test(articles):
    """Very conservative sentiment analysis test"""
    
    print("\n🎯 Conservative Sentiment Analysis Test")
    print("=" * 50)
    
    # Load model on GPU if available, with small batch size
    device = 0 if torch.cuda.is_available() else -1
    print(f"   Loading model on {'GPU' if device == 0 else 'CPU'}...")
    
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
            batch_size=1  # Very conservative - one at a time
        )
        
        print(f"   ✅ Model loaded successfully")
        
        # Test with just a few articles
        test_articles = articles[:5]  # Only test 5 articles
        
        print(f"   📰 Testing {len(test_articles)} articles...")
        
        results = []
        total_time = 0
        
        for i, article in enumerate(test_articles):
            print(f"      Processing article {i+1}/{len(test_articles)}...", end="")
            
            start_time = time.time()
            result = sentiment_analyzer(article[:500])  # Only first 500 chars to be safe
            end_time = time.time()
            
            article_time = end_time - start_time
            total_time += article_time
            results.append(result[0])
            
            print(f" {article_time*1000:.1f}ms ({result[0]['label']})")
            
            # Small delay to prevent overheating
            time.sleep(0.1)
            
            # Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate stats
        avg_time = total_time / len(test_articles)
        articles_per_sec = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"\n   📊 Results:")
        print(f"      Total time: {total_time:.2f}s")
        print(f"      Average per article: {avg_time*1000:.1f}ms")
        print(f"      Speed: {articles_per_sec:.1f} articles/sec")
        
        # Memory check
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"      GPU Memory used: {allocated:.2f} GB")
        
        return articles_per_sec, avg_time
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0, 0

def main():
    print("=" * 60)
    print("🛡️  JustNews V4 CONSERVATIVE Performance Test")
    print("   (System-friendly, won't crash your PC!)")
    print("=" * 60)
    
    # GPU check
    has_gpu, gpu_memory = safe_gpu_test()
    
    if gpu_memory < 4.0:
        print("   ⚠️  GPU memory < 4GB - using extra conservative settings")
    
    # Generate test articles
    print(f"\n📰 Generating realistic test articles...")
    articles = create_sample_news_articles(10)  # Small sample
    
    avg_length = np.mean([len(article) for article in articles])
    print(f"   Count: {len(articles)} articles")
    print(f"   Average length: {avg_length:.0f} characters")
    print(f"   Sample (first 100 chars): {articles[0][:100]}...")
    
    # Conservative sentiment test
    articles_per_sec, avg_time = conservative_sentiment_test(articles)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 CONSERVATIVE PERFORMANCE SUMMARY")
    print("=" * 60)
    
    if articles_per_sec > 0:
        print(f"\n🎯 Sentiment Analysis Performance:")
        print(f"   Speed: {articles_per_sec:.1f} articles/sec")
        print(f"   Time per article: {avg_time*1000:.1f}ms")
        
        # Realistic projection
        projected_hour = articles_per_sec * 3600
        print(f"   Projected hourly throughput: {projected_hour:,.0f} articles")
        
        # V4 target comparison (200-400 articles/sec)
        target_min, target_max = 200, 400
        
        if articles_per_sec >= target_min:
            percentage = (articles_per_sec / target_max) * 100
            print(f"   ✅ MEETS V4 TARGET: {percentage:.1f}% of maximum target")
        elif articles_per_sec >= 50:
            percentage = (articles_per_sec / target_min) * 100
            print(f"   🟡 Good performance: {percentage:.1f}% of minimum target")
        else:
            print(f"   🔶 Baseline established: {articles_per_sec:.1f} articles/sec")
        
        print(f"\n💡 This is a conservative test with:")
        print(f"   - Small batch size (1 article at a time)")
        print(f"   - Only 5 test articles")
        print(f"   - Truncated to 500 characters")
        print(f"   - Built-in delays to prevent overheating")
        print(f"   ➡️  Real performance could be 5-10x higher with optimization!")
    
    else:
        print(f"   ❌ Test failed - check GPU/CUDA setup")
    
    print(f"\n🛡️  System-safe testing complete!")

if __name__ == "__main__":
    main()
