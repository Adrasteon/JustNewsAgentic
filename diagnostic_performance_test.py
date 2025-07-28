#!/usr/bin/env python3
"""
JustNews V4 Diagnostic Performance Test
Smart monitoring to find the crash point without crashing
"""

import torch
import time
import gc
import psutil
import os
from transformers import pipeline
import numpy as np

def monitor_system():
    """Get current system status"""
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated(0) / 1e9
        gpu_cached = torch.cuda.memory_reserved(0) / 1e9
        gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9
    else:
        gpu_allocated = gpu_cached = gpu_free = 0
    
    ram_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    
    return {
        'gpu_allocated': gpu_allocated,
        'gpu_cached': gpu_cached, 
        'gpu_free': gpu_free,
        'ram_usage': ram_usage,
        'cpu_usage': cpu_usage
    }

def safe_batch_test(articles, max_batch_size=64):
    """Incrementally test batch sizes with crash detection"""
    
    print("\n🔍 DIAGNOSTIC Batch Testing (with crash prevention)")
    print("=" * 60)
    
    device = 0 if torch.cuda.is_available() else -1
    
    try:
        print(f"   Loading model...")
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",  # Smaller, safer model
            device=device,
            batch_size=1  # Start conservative
        )
        print(f"   ✅ Model loaded successfully")
        
        # Test articles - start small
        test_articles = articles[:10]  # Only 10 articles to start
        
        results = {}
        safe_batch_sizes = []
        
        # Test incrementally increasing batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            print(f"\n   🧪 Testing batch size: {batch_size}")
            
            # Monitor before test
            before = monitor_system()
            print(f"      Before: GPU {before['gpu_allocated']:.2f}GB allocated, {before['gpu_free']:.2f}GB free")
            
            try:
                # Small warm-up first
                _ = sentiment_analyzer(test_articles[:1])
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                
                # Main test
                start_time = time.time()
                
                for i in range(0, len(test_articles), batch_size):
                    batch = test_articles[i:i+batch_size]
                    
                    # Check memory before each batch
                    current = monitor_system()
                    if current['gpu_free'] < 1.0:  # Less than 1GB free
                        print(f"      ⚠️ GPU memory getting low: {current['gpu_free']:.2f}GB free")
                        break
                    
                    _ = sentiment_analyzer(batch)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                # Monitor after test
                after = monitor_system()
                
                total_time = end_time - start_time
                articles_per_sec = len(test_articles) / total_time if total_time > 0 else 0
                
                results[batch_size] = {
                    'articles_per_sec': articles_per_sec,
                    'total_time': total_time,
                    'gpu_memory_used': after['gpu_allocated'] - before['gpu_allocated']
                }
                
                safe_batch_sizes.append(batch_size)
                
                print(f"      ✅ Success: {articles_per_sec:.1f} articles/sec")
                print(f"      GPU usage: +{after['gpu_allocated'] - before['gpu_allocated']:.3f}GB")
                print(f"      Memory free: {after['gpu_free']:.2f}GB")
                
                # Clean up after each test
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                time.sleep(0.5)  # Brief pause
                
            except Exception as e:
                print(f"      ❌ FAILED at batch size {batch_size}: {str(e)[:100]}...")
                print(f"      This is likely where the crash occurred!")
                break
        
        return results, safe_batch_sizes
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return {}, []

def conservative_scale_test(articles, safe_batch_size):
    """Test with more articles using proven safe batch size"""
    
    print(f"\n📈 SCALE TEST with safe batch size {safe_batch_size}")
    print("=" * 60)
    
    if not safe_batch_size:
        print("   ⚠️ No safe batch size found - skipping scale test")
        return {}
    
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1,
            batch_size=safe_batch_size
        )
        
        # Test with increasing numbers of articles
        article_counts = [10, 20, 30, 50]
        scale_results = {}
        
        for count in article_counts:
            if count > len(articles):
                continue
                
            print(f"   📊 Testing {count} articles...")
            
            test_articles = articles[:count]
            
            before = monitor_system()
            start_time = time.time()
            
            # Process in safe batches
            for i in range(0, len(test_articles), safe_batch_size):
                batch = test_articles[i:i+safe_batch_size]
                _ = sentiment_analyzer(batch)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            after = monitor_system()
            
            total_time = end_time - start_time
            articles_per_sec = len(test_articles) / total_time
            
            scale_results[count] = {
                'articles_per_sec': articles_per_sec,
                'gpu_memory_used': after['gpu_allocated'] - before['gpu_allocated']
            }
            
            print(f"      Speed: {articles_per_sec:.1f} articles/sec")
            print(f"      GPU: +{after['gpu_allocated'] - before['gpu_allocated']:.3f}GB")
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            time.sleep(0.5)
        
        return scale_results
        
    except Exception as e:
        print(f"   ❌ Scale test failed: {e}")
        return {}

def main():
    print("=" * 70)
    print("🔍 JustNews V4 DIAGNOSTIC Performance Test")
    print("   (Finding the crash point safely)")
    print("=" * 70)
    
    # System info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🔥 GPU: {gpu_name}")
        print(f"💾 Total Memory: {gpu_memory:.1f} GB")
    
    initial_status = monitor_system()
    print(f"📊 Initial Status:")
    print(f"   GPU Free: {initial_status['gpu_free']:.2f} GB")
    print(f"   RAM Usage: {initial_status['ram_usage']:.1f}%")
    
    # Generate test articles
    articles = []
    template = "Breaking news from financial markets as technology stocks show significant volatility amid regulatory concerns and market uncertainty. Industry analysts monitor the situation as investors react to new policies impacting growth prospects. The developments come during sensitive times with geopolitical tensions affecting market behavior patterns."
    
    for i in range(50):
        articles.append(f"{template} Article {i+1} provides additional context about recent market developments.")
    
    print(f"\n📰 Generated {len(articles)} test articles (avg {np.mean([len(a) for a in articles]):.0f} chars)")
    
    # Diagnostic batch testing
    batch_results, safe_batches = safe_batch_test(articles)
    
    # Scale testing with safe settings
    if safe_batches:
        best_safe_batch = max(safe_batches)
        scale_results = conservative_scale_test(articles, best_safe_batch)
    else:
        scale_results = {}
    
    # Summary
    print("\n" + "=" * 70)
    print("🔍 DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    if batch_results:
        print(f"\n✅ Safe batch sizes found: {safe_batches}")
        print(f"🚫 Crash point: Likely at batch size {max(safe_batches) * 2} or higher")
        
        print(f"\n📊 Performance by batch size:")
        for batch_size, result in batch_results.items():
            print(f"   Batch {batch_size:2d}: {result['articles_per_sec']:6.1f} articles/sec")
    
    if scale_results:
        print(f"\n📈 Scale test results:")
        best_performance = 0
        for count, result in scale_results.items():
            speed = result['articles_per_sec']
            if speed > best_performance:
                best_performance = speed
            print(f"   {count:2d} articles: {speed:6.1f} articles/sec")
        
        print(f"\n🏆 Best safe performance: {best_performance:.1f} articles/sec")
        
        # V4 target analysis
        target_min = 200
        if best_performance >= target_min:
            print(f"   ✅ MEETS V4 TARGET!")
        else:
            progress = (best_performance / target_min) * 100
            print(f"   📊 Progress: {progress:.1f}% toward V4 target")
    
    print(f"\n💡 Next steps:")
    print(f"   - Use batch size ≤ {max(safe_batches) if safe_batches else 1} for stability")
    print(f"   - Investigate memory management for larger batches")
    print(f"   - Consider model optimization techniques")

if __name__ == "__main__":
    main()
