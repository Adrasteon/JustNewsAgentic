#!/usr/bin/env python3
"""
JustNews V4 Performance-Optimized Test
For water-cooled systems - let's see what your RTX 3090 can really do!
"""

import torch
import time
import gc
from transformers import pipeline
import numpy as np

def create_realistic_news_articles(count=50):
    """Generate more realistic news articles for proper testing"""
    
    templates = [
        "Breaking news from the financial markets today as major technology stocks experienced significant volatility amid concerns about regulatory changes and market uncertainty. Industry analysts are closely monitoring the situation as investors react to new policies that could impact future growth prospects. The developments come at a critical time when market sentiment has been particularly sensitive to geopolitical tensions and economic indicators. Market participants are awaiting further clarification from regulatory bodies about the scope and timeline of these potential changes. Early trading sessions showed mixed results across different sectors, with some defensive stocks gaining ground while growth-oriented investments faced selling pressure. This pattern reflects the broader uncertainty that has characterized recent market behavior. Experts suggest that investors should maintain a diversified approach and focus on long-term fundamentals rather than short-term market fluctuations. The situation continues to evolve as stakeholders assess the potential implications.",
        
        "Climate scientists have announced breakthrough findings that could reshape our understanding of environmental patterns and weather systems across multiple regions. The comprehensive study, conducted over several years, involved data collection from numerous monitoring stations and advanced modeling techniques that provide unprecedented insights into atmospheric behavior. Researchers found that certain phenomena are occurring more frequently than previously predicted, with significant implications for agriculture, urban planning, and disaster preparedness strategies worldwide. The findings suggest that current adaptation strategies may need substantial updates to account for these newly identified patterns. Environmental policy experts are reviewing the research to determine what adjustments might be necessary in current climate action plans and international cooperation frameworks. The study's methodology involved collaboration between multiple institutions and incorporated both satellite data and ground-based observations to create the most comprehensive picture to date.",
        
        "Technology innovation continues to accelerate as companies announce breakthrough developments in artificial intelligence, quantum computing, and biotechnology sectors that promise to transform industries ranging from healthcare and finance to manufacturing and entertainment. These advances represent years of research and development efforts by teams worldwide who are collaborating on projects that could deliver significant benefits to society while addressing some of the most complex challenges facing humanity. The pace of innovation has been particularly notable in areas where interdisciplinary approaches are yielding unexpected results and opening new possibilities for practical applications. Industry leaders emphasize the importance of responsible development and implementation of these technologies to ensure they serve the broader public interest. Regulatory frameworks are being developed to ensure that innovation proceeds safely and ethically while maintaining the momentum needed for continued progress."
    ]
    
    articles = []
    for i in range(count):
        base_article = templates[i % len(templates)]
        variation = f" Article {i+1} Update: Recent developments continue to shape the landscape with new information emerging from ongoing research and analysis efforts by experts in the field."
        articles.append(base_article + variation)
    
    return articles

def optimized_sentiment_test(articles):
    """Performance-optimized sentiment analysis for water-cooled systems"""
    
    print("\n🚀 PERFORMANCE-OPTIMIZED Sentiment Analysis Test")
    print("=" * 60)
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"   Loading model on GPU with optimized settings...")
    
    try:
        # Optimized pipeline settings for performance
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",  # Better model
            device=device,
            batch_size=16,  # Much more aggressive batching
            max_length=512,  # Full article processing
            truncation=True
        )
        
        print(f"   ✅ Model loaded successfully")
        
        # Test different batch sizes to find optimal
        batch_sizes = [1, 4, 8, 16, 32]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n   🔥 Testing batch size: {batch_size}")
            
            # Use subset of articles for each test
            test_articles = articles[:min(25, len(articles))]
            
            # Warm up
            _ = sentiment_analyzer(test_articles[:2])
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            
            for i in range(0, len(test_articles), batch_size):
                batch = test_articles[i:i+batch_size]
                _ = sentiment_analyzer(batch)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_time = end_time - start_time
            articles_per_sec = len(test_articles) / total_time
            avg_time_per_article = total_time / len(test_articles) * 1000
            
            results[batch_size] = {
                'articles_per_sec': articles_per_sec,
                'avg_time_ms': avg_time_per_article,
                'total_time': total_time
            }
            
            print(f"      Time: {total_time:.2f}s")
            print(f"      Speed: {articles_per_sec:.1f} articles/sec")
            print(f"      Avg per article: {avg_time_per_article:.1f}ms")
            
            # Memory check
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e9
                cached = torch.cuda.memory_reserved(0) / 1e9
                print(f"      GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
            
            # Small delay between tests
            time.sleep(1)
            torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {}

def stress_test(articles):
    """Stress test for water-cooled systems"""
    
    print(f"\n❄️ WATER-COOLED STRESS TEST")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("   ⚠️ GPU not available - skipping stress test")
        return {}
    
    try:
        # High-performance configuration
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0,
            batch_size=32,  # Aggressive batching
            max_length=512,
            truncation=True
        )
        
        print(f"   🔥 Running continuous processing for 30 seconds...")
        print(f"   📊 Articles to process: {len(articles)}")
        
        start_time = time.time()
        total_processed = 0
        iteration = 0
        
        while time.time() - start_time < 30:  # 30-second stress test
            iteration += 1
            iter_start = time.time()
            
            # Process all articles in batches
            for i in range(0, len(articles), 32):
                batch = articles[i:i+32]
                _ = sentiment_analyzer(batch)
                total_processed += len(batch)
            
            iter_time = time.time() - iter_start
            articles_per_sec = len(articles) / iter_time
            
            if iteration % 3 == 0:  # Report every 3rd iteration
                allocated = torch.cuda.memory_allocated(0) / 1e9
                cached = torch.cuda.memory_reserved(0) / 1e9
                elapsed = time.time() - start_time
                print(f"      Iteration {iteration}: {articles_per_sec:.1f} art/sec, {allocated:.2f}GB used, {elapsed:.1f}s elapsed")
        
        total_time = time.time() - start_time
        overall_speed = total_processed / total_time
        
        print(f"\n   🏆 Stress Test Results:")
        print(f"      Duration: {total_time:.1f} seconds")
        print(f"      Total processed: {total_processed:,} articles")
        print(f"      Sustained speed: {overall_speed:.1f} articles/sec")
        print(f"      Iterations completed: {iteration}")
        
        return {
            'sustained_speed': overall_speed,
            'total_processed': total_processed,
            'iterations': iteration
        }
        
    except Exception as e:
        print(f"   ❌ Stress test error: {e}")
        return {}

def main():
    print("=" * 70)
    print("🚀 JustNews V4 PERFORMANCE-OPTIMIZED Test")
    print("   (Water-cooled systems - let's push this RTX 3090!)")
    print("=" * 70)
    
    # GPU status
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🔥 GPU: {gpu_name}")
        print(f"💾 Memory: {gpu_memory:.1f} GB")
    
    # Generate more articles for proper testing
    print(f"\n📰 Generating realistic news articles...")
    articles = create_realistic_news_articles(50)
    
    avg_length = np.mean([len(article) for article in articles])
    print(f"   Count: {len(articles)} articles")
    print(f"   Average length: {avg_length:.0f} characters")
    
    # Optimized batch testing
    batch_results = optimized_sentiment_test(articles)
    
    # Stress test for water-cooled systems
    stress_results = stress_test(articles)
    
    # Final summary
    print("\n" + "=" * 70)
    print("🏆 PERFORMANCE-OPTIMIZED RESULTS")
    print("=" * 70)
    
    if batch_results:
        print(f"\n🎯 Batch Size Performance:")
        best_performance = 0
        best_batch = 1
        
        for batch_size, result in batch_results.items():
            speed = result['articles_per_sec']
            if speed > best_performance:
                best_performance = speed
                best_batch = batch_size
            
            print(f"   Batch {batch_size:2d}: {speed:6.1f} articles/sec ({result['avg_time_ms']:5.1f}ms per article)")
        
        print(f"\n🏆 Best Performance: {best_performance:.1f} articles/sec (batch size {best_batch})")
        
        # V4 target comparison
        target_min, target_max = 200, 400
        
        if best_performance >= target_min:
            percentage = (best_performance / target_max) * 100
            print(f"   ✅ EXCEEDS V4 TARGET: {percentage:.1f}% of maximum target")
        else:
            percentage = (best_performance / target_min) * 100
            print(f"   🟡 Progress toward target: {percentage:.1f}% of minimum target")
    
    if stress_results:
        sustained = stress_results['sustained_speed']
        print(f"\n❄️ Water-Cooled Stress Test:")
        print(f"   Sustained speed: {sustained:.1f} articles/sec over 30 seconds")
        print(f"   Total processed: {stress_results['total_processed']:,} articles")
        print(f"   System stability: {'✅ Excellent' if sustained > 20 else '🟡 Good' if sustained > 10 else '🔶 Baseline'}")
    
    print(f"\n🚀 Your water-cooled RTX 3090 is ready for serious workloads!")

if __name__ == "__main__":
    main()
