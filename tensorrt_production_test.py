#!/usr/bin/env python3
"""
Production TensorRT Performance Test
Tests native TensorRT engines with realistic workloads targeting 300-600 articles/sec
"""

import time
import logging
from agents.analyst.tensorrt_acceleration import TensorRTAnalyst

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_articles(count: int = 100):
    """Generate realistic test articles for performance testing"""
    articles = []
    
    # Realistic news article templates
    templates = [
        "Breaking news: The stock market showed significant gains today with technology companies leading the surge. Investors are optimistic about upcoming earnings reports and regulatory changes that could benefit the sector.",
        "In a surprising turn of events, the presidential election results have been announced with voter turnout reaching record highs. Political analysts suggest this outcome could reshape domestic and foreign policy for years to come.",
        "Scientists at the university have made a groundbreaking discovery in renewable energy technology. The new solar panel design promises to increase efficiency by 40% while reducing manufacturing costs significantly.",
        "The city council voted unanimously to approve the new infrastructure project that will modernize public transportation systems. Construction is expected to begin next year and create thousands of jobs.",
        "Economic indicators suggest inflation rates may be stabilizing after months of uncertainty. Federal Reserve officials are cautiously optimistic about the current monetary policy effectiveness.",
        "A major cybersecurity breach at a prominent financial institution has exposed sensitive customer data. Security experts warn this incident highlights growing vulnerabilities in digital banking systems.",
        "The climate summit concluded with ambitious commitments from world leaders to reduce carbon emissions. Environmental groups praise the agreements while critics question implementation feasibility.",
        "Sports fans celebrate as the local team advances to the championship finals after an incredible season. The victory marks the franchise's first playoff appearance in over a decade."
    ]
    
    for i in range(count):
        template_idx = i % len(templates)
        article = f"[Article {i+1}] {templates[template_idx]}"
        # Add some variation to lengths
        if i % 3 == 0:
            article += " Additional context and analysis from industry experts suggests this development could have far-reaching implications for the sector."
        articles.append(article)
    
    return articles

def test_production_performance():
    """Test TensorRT acceleration with production-scale workloads"""
    print("🚀 PRODUCTION TENSORRT PERFORMANCE TEST")
    print("=" * 60)
    
    # Initialize TensorRT accelerator
    accelerator = TensorRTAnalyst()
    
    # Test configurations
    test_configs = [
        {"batch_size": 1, "description": "Single Article Processing"},
        {"batch_size": 10, "description": "Small Batch Processing"},
        {"batch_size": 25, "description": "Medium Batch Processing"},
        {"batch_size": 50, "description": "Large Batch Processing"},
        {"batch_size": 100, "description": "Maximum Batch Processing"}
    ]
    
    results = {}
    
    for config in test_configs:
        batch_size = config["batch_size"]
        description = config["description"]
        
        print(f"\n📊 Testing {description} ({batch_size} articles)")
        print("-" * 40)
        
        # Generate test articles
        articles = generate_test_articles(batch_size)
        
        # Test sentiment analysis
        start_time = time.time()
        if batch_size == 1:
            sentiment_score = accelerator.score_sentiment_tensorrt(articles[0])
            sentiment_results = [sentiment_score]
        else:
            sentiment_results = accelerator.score_sentiment_batch_tensorrt(articles)
        
        sentiment_time = time.time() - start_time
        sentiment_rate = batch_size / sentiment_time if sentiment_time > 0 else 0
        
        # Test bias analysis
        start_time = time.time()
        if batch_size == 1:
            bias_score = accelerator.score_bias_tensorrt(articles[0])
            bias_results = [bias_score]
        else:
            bias_results = accelerator.score_bias_batch_tensorrt(articles)
        
        bias_time = time.time() - start_time
        bias_rate = batch_size / bias_time if bias_time > 0 else 0
        
        # Store results
        results[batch_size] = {
            'sentiment_rate': sentiment_rate,
            'bias_rate': bias_rate,
            'sentiment_time': sentiment_time,
            'bias_time': bias_time,
            'total_rate': batch_size / (sentiment_time + bias_time) if (sentiment_time + bias_time) > 0 else 0
        }
        
        print(f"✅ Sentiment: {sentiment_rate:.1f} articles/sec ({sentiment_time:.3f}s)")
        print(f"✅ Bias: {bias_rate:.1f} articles/sec ({bias_time:.3f}s)")
        print(f"🎯 Combined: {results[batch_size]['total_rate']:.1f} articles/sec")
        
        # Sample results
        if sentiment_results and bias_results and sentiment_results[0] is not None and bias_results[0] is not None:
            print(f"📄 Sample Results:")
            print(f"   Sentiment: {sentiment_results[0]:.3f}")
            print(f"   Bias: {bias_results[0]:.3f}")
        else:
            print(f"📄 Sample Results:")
            print(f"   ⚠️  Some results were None - engines may have failed")
    
    # Performance summary
    print(f"\n🎯 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    best_sentiment = max(results.values(), key=lambda x: x['sentiment_rate'])
    best_bias = max(results.values(), key=lambda x: x['bias_rate'])
    best_combined = max(results.values(), key=lambda x: x['total_rate'])
    
    print(f"🥇 Peak Sentiment Performance: {best_sentiment['sentiment_rate']:.1f} articles/sec")
    print(f"🥇 Peak Bias Performance: {best_bias['bias_rate']:.1f} articles/sec")
    print(f"🥇 Peak Combined Performance: {best_combined['total_rate']:.1f} articles/sec")
    
    # Performance targets
    baseline = 151.4  # articles/sec
    target_2x = baseline * 2  # 302.8 articles/sec
    target_4x = baseline * 4  # 605.6 articles/sec
    
    peak_performance = best_combined['total_rate']
    improvement_factor = peak_performance / baseline
    
    print(f"\n🎯 PERFORMANCE ANALYSIS")
    print("-" * 30)
    print(f"Baseline (HuggingFace): {baseline:.1f} articles/sec")
    print(f"Current Performance: {peak_performance:.1f} articles/sec")
    print(f"Improvement Factor: {improvement_factor:.2f}x")
    
    if improvement_factor >= 4.0:
        print("🚀 EXCEPTIONAL: Achieved 4x+ performance target!")
    elif improvement_factor >= 2.0:
        print("✅ SUCCESS: Achieved 2x performance target!")
    elif improvement_factor >= 1.5:
        print("⚡ GOOD: Significant performance improvement")
    else:
        print("⚠️  OPTIMIZING: Below 1.5x target, needs optimization")
    
    # Get final performance stats
    stats = accelerator.get_performance_stats()
    print(f"\n📊 System Statistics:")
    print(f"   Total Requests: {stats.get('trt_requests', 0) + stats.get('fallback_requests', 0)}")
    print(f"   TensorRT Requests: {stats.get('trt_requests', 0)}")
    print(f"   Fallback Requests: {stats.get('fallback_requests', 0)}")
    print(f"   Average Processing Time: {stats.get('trt_avg_time', 0)*1000:.1f}ms")
    
    return results

if __name__ == "__main__":
    try:
        test_production_performance()
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
