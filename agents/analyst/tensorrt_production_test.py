#!/usr/bin/env python3
"""
TensorRT Production Stress Test for JustNews V4 Analyst
Validates TensorRT acceleration under production load conditions

Performance Targets:
- Individual: 300-600 articles/sec (2-4x over 151.4 baseline)
- Batch: 400-800 articles/sec with optimized batching
- Memory: <6GB VRAM with FP16 precision
- Stability: Zero crashes under sustained load
"""

import logging
import os
import statistics
import sys
import time
from typing import Any, Dict, List

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tensorrt_acceleration import TensorRTAnalyst

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorRTProductionValidator:
    """Production validation suite for TensorRT optimization"""
    
    def __init__(self):
        self.analyst = TensorRTAnalyst()
        self.test_articles = self._generate_test_articles()
        self.results = {}
        
    def _generate_test_articles(self) -> List[str]:
        """Generate realistic news articles for testing"""
        articles = [
            "Breaking news: Technology stocks surge as AI companies report record quarterly earnings, driving market optimism and investor confidence to new heights.",
            "Economic indicators show steady growth in the manufacturing sector with employment rates climbing for the third consecutive month, signaling robust recovery.",
            "Weather forecast predicts severe storms across the midwest region this weekend, with emergency services preparing for potential flooding and power outages.",
            "Sports update: Championship finals set for this weekend as two top-ranked teams prepare for what analysts call the most anticipated match of the season.",
            "Healthcare breakthrough announced as researchers develop new treatment protocol showing promising results in clinical trials for chronic disease management.",
            "Environmental protection measures gain bipartisan support in congress as lawmakers address climate change concerns with comprehensive policy proposals.",
            "Educational reforms focus on digital literacy programs designed to prepare students for technology-driven careers in the rapidly evolving job market.",
            "Transportation infrastructure improvements include high-speed rail expansion and electric vehicle charging network development across major urban centers.",
            "Cultural events celebrate diversity and community engagement through art exhibitions, music festivals, and international food celebrations throughout the city.",
            "Innovation in renewable energy continues with new solar panel efficiency records and wind turbine technology advancing sustainable power generation capabilities.",
            "Financial markets respond positively to interest rate decisions while analysts monitor inflation trends and economic policy impacts on consumer spending.",
            "Scientific research reveals new discoveries in quantum computing applications with potential implications for cybersecurity and data processing capabilities.",
            "Local government announces urban development projects aimed at creating affordable housing and improving public transportation accessibility for residents.",
            "International trade agreements strengthen economic partnerships between nations while addressing supply chain challenges and promoting fair trade practices.",
            "Healthcare workers receive recognition for their dedication during challenging times as medical facilities expand services and improve patient care quality.",
            "Technology companies invest heavily in artificial intelligence research and development to create innovative solutions for business automation and efficiency.",
            "Environmental conservation efforts protect endangered species through habitat restoration programs and community-based wildlife protection initiatives across the region.",
            "Education technology transforms classroom learning with interactive digital tools and personalized learning platforms designed to enhance student engagement and achievement.",
            "Energy sector developments include nuclear power modernization and battery storage technology improvements supporting grid stability and renewable energy integration.",
            "Social media platforms implement new privacy features and content moderation policies to address user safety concerns and regulatory compliance requirements."
        ]
        
        # Extend to 100 articles by cycling through with variations
        extended_articles = []
        for i in range(100):
            base_article = articles[i % len(articles)]
            # Add slight variations to simulate real news diversity
            if i >= len(articles):
                base_article = f"Update: {base_article}"
            if i >= len(articles) * 2:
                base_article = f"Analysis: {base_article}"
            extended_articles.append(base_article)
            
        return extended_articles
    
    def test_individual_performance(self, num_tests: int = 50) -> Dict[str, Any]:
        """Test individual article processing performance"""
        logger.info(f"🧪 Testing individual performance ({num_tests} articles)")
        
        sentiment_times = []
        bias_times = []
        
        for i in range(num_tests):
            article = self.test_articles[i % len(self.test_articles)]
            
            # Test sentiment
            start_time = time.time()
            sentiment = self.analyst.score_sentiment_tensorrt(article)
            sentiment_time = time.time() - start_time
            sentiment_times.append(sentiment_time)
            
            # Test bias
            start_time = time.time()
            bias = self.analyst.score_bias_tensorrt(article)
            bias_time = time.time() - start_time
            bias_times.append(bias_time)
            
            if i % 10 == 0:
                logger.info(f"  Progress: {i+1}/{num_tests} articles processed")
        
        results = {
            'num_articles': num_tests,
            'sentiment_avg_time': statistics.mean(sentiment_times),
            'sentiment_articles_per_sec': num_tests / sum(sentiment_times),
            'bias_avg_time': statistics.mean(bias_times),
            'bias_articles_per_sec': num_tests / sum(bias_times),
            'total_time': sum(sentiment_times) + sum(bias_times)
        }
        
        logger.info(f"✅ Individual test complete:")
        logger.info(f"  Sentiment: {results['sentiment_articles_per_sec']:.1f} articles/sec")
        logger.info(f"  Bias: {results['bias_articles_per_sec']:.1f} articles/sec")
        
        return results
    
    def test_batch_performance(self, batch_sizes: List[int] = [10, 25, 50, 100]) -> Dict[str, Any]:
        """Test batch processing performance with various sizes"""
        logger.info(f"🚀 Testing batch performance (sizes: {batch_sizes})")
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"  Testing batch size: {batch_size}")
            
            # Prepare batch
            batch_articles = self.test_articles[:batch_size]
            
            # Test sentiment batch
            start_time = time.time()
            sentiment_results = self.analyst.score_sentiment_batch_tensorrt(batch_articles)
            sentiment_time = time.time() - start_time
            
            # Test bias batch
            start_time = time.time()
            bias_results = self.analyst.score_bias_batch_tensorrt(batch_articles)
            bias_time = time.time() - start_time
            
            batch_results[batch_size] = {
                'sentiment_time': sentiment_time,
                'sentiment_articles_per_sec': batch_size / sentiment_time,
                'bias_time': bias_time,
                'bias_articles_per_sec': batch_size / bias_time,
                'sentiment_success_rate': sum(1 for r in sentiment_results if r is not None) / len(sentiment_results),
                'bias_success_rate': sum(1 for r in bias_results if r is not None) / len(bias_results)
            }
            
            logger.info(f"    Sentiment: {batch_results[batch_size]['sentiment_articles_per_sec']:.1f} articles/sec")
            logger.info(f"    Bias: {batch_results[batch_size]['bias_articles_per_sec']:.1f} articles/sec")
        
        return batch_results
    
    def test_sustained_load(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Test sustained load performance"""
        logger.info(f"⏱️  Testing sustained load ({duration_minutes} minutes)")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        total_articles = 0
        error_count = 0
        performance_samples = []
        
        while time.time() < end_time:
            batch_articles = self.test_articles[:25]  # Use 25-article batches
            
            batch_start = time.time()
            
            try:
                sentiment_results = self.analyst.score_sentiment_batch_tensorrt(batch_articles)
                bias_results = self.analyst.score_bias_batch_tensorrt(batch_articles)
                
                batch_time = time.time() - batch_start
                articles_per_sec = len(batch_articles) * 2 / batch_time  # Both sentiment and bias
                performance_samples.append(articles_per_sec)
                
                total_articles += len(batch_articles) * 2
                
            except Exception as e:
                error_count += 1
                logger.error(f"❌ Batch processing error: {e}")
            
            # Brief pause to simulate realistic load
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        
        results = {
            'duration_minutes': duration_minutes,
            'total_articles': total_articles,
            'avg_articles_per_sec': total_articles / total_time,
            'peak_articles_per_sec': max(performance_samples) if performance_samples else 0,
            'error_count': error_count,
            'error_rate': error_count / (total_articles / 50) if total_articles > 0 else 0,  # Errors per batch
            'performance_stability': statistics.stdev(performance_samples) if len(performance_samples) > 1 else 0
        }
        
        logger.info(f"✅ Sustained load test complete:")
        logger.info(f"  Average: {results['avg_articles_per_sec']:.1f} articles/sec")
        logger.info(f"  Peak: {results['peak_articles_per_sec']:.1f} articles/sec")
        logger.info(f"  Errors: {error_count}")
        
        return results
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete TensorRT validation suite"""
        logger.info("🎯 Starting TensorRT Production Validation")
        
        # Check system status
        if not self.analyst.trt_available or not self.analyst.trt_engines:
            logger.error("❌ TensorRT engines not available - run with --build-engines first")
            return {}
        
        logger.info(f"✅ TensorRT engines active: {list(self.analyst.trt_engines.keys())}")
        
        results = {
            'system_info': {
                'tensorrt_engines': list(self.analyst.trt_engines.keys()),
                'tensorrt_available': self.analyst.trt_available,
                'baseline_performance': 151.4  # articles/sec
            }
        }
        
        # Run tests
        results['individual_performance'] = self.test_individual_performance(50)
        results['batch_performance'] = self.test_batch_performance([10, 25, 50, 100])
        results['sustained_load'] = self.test_sustained_load(3)  # 3-minute test
        
        # Calculate overall metrics
        best_sentiment = max(
            results['individual_performance']['sentiment_articles_per_sec'],
            max(batch['sentiment_articles_per_sec'] for batch in results['batch_performance'].values())
        )
        
        best_bias = max(
            results['individual_performance']['bias_articles_per_sec'],
            max(batch['bias_articles_per_sec'] for batch in results['batch_performance'].values())
        )
        
        baseline = results['system_info']['baseline_performance']
        sentiment_improvement = best_sentiment / baseline
        bias_improvement = best_bias / baseline
        
        results['summary'] = {
            'best_sentiment_performance': best_sentiment,
            'best_bias_performance': best_bias,
            'sentiment_improvement': sentiment_improvement,
            'bias_improvement': bias_improvement,
            'target_achieved': sentiment_improvement >= 2.0 and bias_improvement >= 2.0,
            'stability_score': 1.0 - (results['sustained_load']['error_rate'] / 100)
        }
        
        # Display results
        logger.info("📊 TENSORRT VALIDATION RESULTS:")
        logger.info(f"  Best Sentiment: {best_sentiment:.1f} articles/sec ({sentiment_improvement:.2f}x baseline)")
        logger.info(f"  Best Bias: {best_bias:.1f} articles/sec ({bias_improvement:.2f}x baseline)")
        logger.info(f"  Target Achieved: {'✅ YES' if results['summary']['target_achieved'] else '⚠️  Not yet'}")
        logger.info(f"  Stability: {results['summary']['stability_score']:.3f}")
        
        if results['summary']['target_achieved']:
            logger.info("🎉 SUCCESS: TensorRT optimization targets achieved!")
        else:
            logger.info("🔧 OPTIMIZATION NEEDED: Continue TensorRT engine development")
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TensorRT Production Validation")
    parser.add_argument("--quick", action="store_true", help="Run quick validation (1 minute)")
    parser.add_argument("--full", action="store_true", help="Run full validation suite")
    
    args = parser.parse_args()
    
    validator = TensorRTProductionValidator()
    
    if args.quick:
        logger.info("🏃 Running quick validation")
        results = {
            'individual': validator.test_individual_performance(20),
            'batch': validator.test_batch_performance([25]),
            'load': validator.test_sustained_load(1)
        }
    elif args.full:
        results = validator.run_full_validation()
    else:
        # Default test
        logger.info("🧪 Running standard TensorRT validation")
        results = validator.run_full_validation()
    
    logger.info("✅ TensorRT validation complete!")
