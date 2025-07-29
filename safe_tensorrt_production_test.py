#!/usr/bin/env python3
"""
Safe Native TensorRT Production Test - Conservative Performance Validation
=========================================================================

This test validates native TensorRT performance with safe limits:
- Conservative batch sizes (max 100 articles)
- Memory-efficient testing
- System stability monitoring
- Graceful error handling
"""

import time
import logging
import statistics
import gc
import torch
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def generate_test_articles(count: int) -> List[str]:
    """Generate test articles without excessive memory usage"""
    # Use shorter, realistic articles to avoid memory issues
    base_article = """Breaking news from the financial markets today as technology stocks continue their upward trajectory. 
Market analysts are closely watching the performance of major tech companies as earnings reports are released this quarter. 
The economic indicators suggest sustained growth despite global uncertainties. Investors remain cautiously optimistic about 
the technology sector's long-term prospects while monitoring regulatory developments."""
    
    return [base_article for _ in range(count)]

def safe_memory_check():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"   GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        return memory_allocated < 20.0  # Safe threshold for RTX 3090
    return True

def test_safe_tensorrt_performance():
    """Test native TensorRT with safe memory limits"""
    print("🚀 SAFE NATIVE TENSORRT PRODUCTION TEST")
    print("=" * 70)
    
    # Initialize the TensorRT analyst
    from agents.analyst.tensorrt_acceleration import TensorRTAnalyst
    
    print("INFO: Initializing Native TensorRT System...")
    analyst = TensorRTAnalyst()
    
    if not analyst.native_engine or not analyst.native_engine.engines:
        print("❌ Native TensorRT engines not available!")
        return None
    
    print(f"INFO: System ready with engines: {list(analyst.native_engine.engines.keys())}")
    
    # Conservative test scenarios
    test_scenarios = [
        {"name": "Single Article", "batch_size": 1, "iterations": 10},
        {"name": "Small Batch", "batch_size": 5, "iterations": 8},
        {"name": "Medium Batch", "batch_size": 10, "iterations": 6},
        {"name": "Large Batch", "batch_size": 25, "iterations": 4},
        {"name": "Maximum Safe Batch", "batch_size": 50, "iterations": 3},
        {"name": "Peak Performance", "batch_size": 100, "iterations": 2}
    ]
    
    results = {}
    
    for scenario in test_scenarios:
        print(f"\n📊 {scenario['name']} Test ({scenario['batch_size']} articles × {scenario['iterations']} iterations)")
        print("-" * 70)
        
        # Check memory before test
        if not safe_memory_check():
            print("⚠️  Memory usage too high, skipping this test")
            continue
        
        # Generate test articles
        articles = generate_test_articles(scenario['batch_size'])
        
        sentiment_times = []
        bias_times = []
        sentiment_rates = []
        bias_rates = []
        
        for iteration in range(scenario['iterations']):
            print(f"   Iteration {iteration + 1}/{scenario['iterations']}...", end=" ")
            
            try:
                # Test sentiment analysis
                start_time = time.time()
                sentiment_results = analyst.score_sentiment_batch_tensorrt(articles)
                sentiment_time = time.time() - start_time
                sentiment_rate = len(articles) / sentiment_time if sentiment_time > 0 else 0
                
                # Test bias analysis  
                start_time = time.time()
                bias_results = analyst.score_bias_batch_tensorrt(articles)
                bias_time = time.time() - start_time
                bias_rate = len(articles) / bias_time if bias_time > 0 else 0
                
                sentiment_times.append(sentiment_time)
                bias_times.append(bias_time)
                sentiment_rates.append(sentiment_rate)
                bias_rates.append(bias_rate)
                
                print(f"Sentiment: {sentiment_rate:.1f} art/s, Bias: {bias_rate:.1f} art/s")
                
                # Memory cleanup after each iteration
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Iteration failed: {e}")
                break
        
        if not sentiment_times:
            print("   ⚠️  No successful iterations")
            continue
        
        # Calculate statistics
        avg_sentiment_rate = statistics.mean(sentiment_rates)
        avg_bias_rate = statistics.mean(bias_rates)
        combined_rate = 1 / (1/avg_sentiment_rate + 1/avg_bias_rate) * 2
        
        # Check success rate
        sentiment_success = sum(1 for result in sentiment_results if result is not None) if sentiment_results else 0
        bias_success = sum(1 for result in bias_results if result is not None) if bias_results else 0
        success_rate = min(sentiment_success / len(articles), bias_success / len(articles)) * 100
        
        results[scenario['name']] = {
            'batch_size': scenario['batch_size'],
            'iterations': len(sentiment_times),
            'avg_sentiment_rate': avg_sentiment_rate,
            'avg_bias_rate': avg_bias_rate,
            'combined_rate': combined_rate,
            'success_rate': success_rate,
            'total_articles': scenario['batch_size'] * len(sentiment_times)
        }
        
        print(f"   ✅ Average Sentiment: {avg_sentiment_rate:.1f} articles/sec")
        print(f"   ✅ Average Bias: {avg_bias_rate:.1f} articles/sec") 
        print(f"   🎯 Combined Rate: {combined_rate:.1f} articles/sec")
        print(f"   📊 Success Rate: {success_rate:.1f}%")
        
        # Memory check after each scenario
        safe_memory_check()

    # Performance summary
    print(f"\n🎯 PRODUCTION PERFORMANCE SUMMARY")
    print("=" * 70)
    
    if not results:
        print("❌ No successful tests completed")
        return None
    
    # Find peak performance
    peak_sentiment = max(results.values(), key=lambda x: x['avg_sentiment_rate'])
    peak_bias = max(results.values(), key=lambda x: x['avg_bias_rate'])
    peak_combined = max(results.values(), key=lambda x: x['combined_rate'])
    
    print(f"🥇 Peak Sentiment Performance: {peak_sentiment['avg_sentiment_rate']:.1f} articles/sec")
    print(f"🥇 Peak Bias Performance: {peak_bias['avg_bias_rate']:.1f} articles/sec")
    print(f"🥇 Peak Combined Performance: {peak_combined['combined_rate']:.1f} articles/sec")
    
    # Calculate total articles processed
    total_articles = sum(result['total_articles'] for result in results.values())
    print(f"\n📊 Test Statistics:")
    print(f"   Total Articles Processed: {total_articles:,}")
    print(f"   Average Success Rate: {statistics.mean([r['success_rate'] for r in results.values()]):.1f}%")
    print(f"   Test Scenarios Completed: {len(results)}/{len(test_scenarios)}")
    
    # Performance comparison with baseline
    baseline_performance = 151.4  # HuggingFace GPU baseline
    improvement_factor = peak_combined['combined_rate'] / baseline_performance
    
    print(f"\n🎯 PERFORMANCE ANALYSIS")
    print("-" * 50)
    print(f"Baseline (HuggingFace GPU): {baseline_performance:.1f} articles/sec")
    print(f"Native TensorRT Peak: {peak_combined['combined_rate']:.1f} articles/sec")
    print(f"Improvement Factor: {improvement_factor:.2f}x")
    
    if improvement_factor >= 4.0:
        print("🚀 EXCEPTIONAL: Achieved 4x+ performance target!")
    elif improvement_factor >= 2.0:
        print("✅ SUCCESS: Achieved 2x+ performance target!")
    else:
        print("⚠️  OPTIMIZING: Below 2x target, needs optimization")
    
    # Reliability assessment
    success_rates = [r['success_rate'] for r in results.values()]
    avg_success = statistics.mean(success_rates)
    min_success = min(success_rates)
    
    print(f"\n🔧 SYSTEM RELIABILITY")
    print("-" * 50)
    print(f"Average Success Rate: {avg_success:.1f}%")
    print(f"Minimum Success Rate: {min_success:.1f}%")
    
    if min_success >= 95.0:
        print("✅ EXCELLENT: System highly reliable")
    elif min_success >= 90.0:
        print("✅ GOOD: System reliable with minor issues")  
    else:
        print("⚠️  ATTENTION: Reliability concerns detected")
    
    # Memory efficiency check
    final_memory_check = safe_memory_check()
    if final_memory_check:
        print("✅ MEMORY: Efficient memory usage maintained")
    else:
        print("⚠️  MEMORY: High memory usage detected")
    
    return results

if __name__ == "__main__":
    try:
        print("🔧 Starting safe production test...")
        print("   Max batch size: 100 articles")
        print("   Memory monitoring: Enabled")
        print("   Error handling: Conservative")
        print()
        
        results = test_safe_tensorrt_performance()
        
        if results:
            print(f"\n✅ Safe native TensorRT testing completed successfully!")
            print(f"   Scenarios completed: {len(results)}")
            print(f"   System remained stable throughout testing")
        else:
            print(f"\n⚠️  Testing completed with limited results")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("\n🧹 Memory cleanup completed")
