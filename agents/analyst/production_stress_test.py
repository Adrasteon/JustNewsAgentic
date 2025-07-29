#!/usr/bin/env python3
"""
Production Stress Test for Native TensorRT Analyst Agent
Tests with realistic article volumes and sizes (1000 articles × 2000 chars each)
"""

import sys
import os
import logging
import time
import requests
import json
import random
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_realistic_article(length: int = 2000) -> str:
    """Generate a realistic news article of specified length"""
    
    # Base article templates with realistic news content
    templates = [
        "Breaking news from the technology sector reveals significant developments in artificial intelligence and machine learning. Industry experts are analyzing the implications of new algorithms that promise to revolutionize data processing capabilities across multiple industries. The advancement represents a major breakthrough in computational efficiency and could transform how businesses approach complex problem-solving. Leading researchers from top universities have collaborated with technology companies to develop these innovative solutions. The new systems demonstrate unprecedented performance improvements over existing methodologies. Market analysts predict substantial economic impact as organizations begin implementing these cutting-edge technologies. Consumer applications are expected to benefit from enhanced processing speeds and improved accuracy in automated systems. The research team published their findings in peer-reviewed journals, highlighting the scientific rigor behind their approach. Initial testing phases showed remarkable results across diverse use cases, from financial modeling to healthcare diagnostics. The technology addresses longstanding challenges in computational complexity while maintaining security and reliability standards. Industry partnerships are forming to accelerate deployment of these innovations across global markets. Regulatory bodies are reviewing the implications to ensure responsible implementation of these powerful new capabilities.",
        
        "Economic indicators suggest significant shifts in global market dynamics as international trade patterns evolve in response to technological innovations and policy changes. Financial institutions are adapting their strategies to accommodate emerging digital currencies and blockchain technologies that are reshaping traditional banking operations. Central banks worldwide are monitoring inflation trends and adjusting monetary policies to maintain economic stability. Investment portfolios are being restructured to include sustainable energy assets and environmental, social, and governance compliant securities. Corporate earnings reports indicate strong performance in technology and healthcare sectors while traditional manufacturing faces transformation challenges. Supply chain disruptions continue to influence pricing strategies across retail and consumer goods industries. Labor markets show evolving skill requirements as automation and artificial intelligence reshape job categories and career paths. Real estate markets reflect changing work patterns with increased demand for flexible commercial spaces and residential properties in suburban areas. Energy sector investments are shifting toward renewable sources as governments implement carbon reduction initiatives and environmental regulations. International trade agreements are being renegotiated to address digital commerce and intellectual property protections in the modern economy.",
        
        "Healthcare breakthroughs in personalized medicine are transforming patient treatment approaches through advanced genetic analysis and targeted therapeutic interventions. Medical researchers have identified new biomarkers that enable earlier detection of diseases and more precise treatment protocols. Clinical trials demonstrate improved patient outcomes when treatments are customized based on individual genetic profiles and molecular characteristics. Pharmaceutical companies are investing heavily in precision medicine technologies that promise to revolutionize drug development and delivery systems. Telemedicine applications are expanding access to specialized care in rural and underserved communities through high-quality video consultations and remote monitoring devices. Electronic health records integration is improving care coordination among healthcare providers while maintaining strict patient privacy and data security standards. Artificial intelligence applications in medical diagnostics are helping physicians identify patterns and anomalies in medical imaging with unprecedented accuracy and speed. Public health initiatives are leveraging big data analytics to track disease outbreaks and implement targeted prevention strategies. Medical device innovations are enabling minimally invasive procedures that reduce recovery times and improve surgical outcomes for patients worldwide.",
    ]
    
    # Select a random template and extend it to reach target length
    base_article = random.choice(templates)
    
    # Additional content to extend articles
    extensions = [
        " Furthermore, industry stakeholders are closely monitoring developments to assess long-term implications for their respective sectors.",
        " Research findings suggest that implementation timelines may vary significantly depending on organizational readiness and resource allocation.",
        " Experts recommend careful evaluation of potential benefits and risks before adopting new technologies or processes.",
        " Collaborative efforts between academic institutions and private sector organizations are accelerating innovation cycles.",
        " Regulatory frameworks are being updated to address emerging challenges and ensure public safety standards are maintained.",
        " Consumer adoption rates indicate strong interest in solutions that provide tangible value and improved user experiences.",
        " International cooperation is essential for addressing global challenges that require coordinated responses across borders.",
        " Sustainability considerations are becoming increasingly important in decision-making processes across all industries.",
        " Educational institutions are updating curricula to prepare students for evolving workplace requirements and skill demands.",
        " Data privacy and security measures are being strengthened to protect sensitive information in digital environments.",
    ]
    
    # Build article to target length
    article = base_article
    while len(article) < length:
        article += " " + random.choice(extensions)
    
    # Trim to exact length if needed
    return article[:length]

def generate_test_dataset(num_articles: int = 1000, avg_length: int = 2000) -> List[str]:
    """Generate a dataset of realistic news articles"""
    logger.info(f"🔄 Generating {num_articles} articles with average length {avg_length} characters...")
    
    articles = []
    for i in range(num_articles):
        # Vary article length slightly (±20%)
        length = random.randint(int(avg_length * 0.8), int(avg_length * 1.2))
        article = generate_realistic_article(length)
        articles.append(article)
        
        if (i + 1) % 100 == 0:
            logger.info(f"   Generated {i + 1}/{num_articles} articles...")
    
    total_chars = sum(len(article) for article in articles)
    avg_actual = total_chars / len(articles)
    
    logger.info(f"✅ Dataset generated: {num_articles} articles, {total_chars:,} total chars, {avg_actual:.0f} avg chars")
    return articles

def test_batch_performance(articles: List[str], batch_size: int = 32, base_url: str = "http://localhost:8004") -> Dict:
    """Test batch processing performance with realistic data"""
    logger.info(f"🚀 Starting production stress test: {len(articles)} articles in batches of {batch_size}")
    
    results = {
        'total_articles': len(articles),
        'batch_size': batch_size,
        'sentiment_results': [],
        'bias_results': [],
        'sentiment_throughput': 0,
        'bias_throughput': 0,
        'total_time': 0,
        'errors': []
    }
    
    # Process in batches
    num_batches = (len(articles) + batch_size - 1) // batch_size
    logger.info(f"📊 Processing {num_batches} batches...")
    
    sentiment_times = []
    bias_times = []
    
    start_total = time.time()
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(articles))
        batch = articles[start_idx:end_idx]
        
        logger.info(f"🔄 Processing batch {batch_idx + 1}/{num_batches} ({len(batch)} articles)...")
        
        # Test sentiment batch
        try:
            payload = {"args": [batch], "kwargs": {}}
            
            start_time = time.time()
            response = requests.post(f"{base_url}/score_sentiment_batch", json=payload, timeout=60)
            end_time = time.time()
            
            if response.status_code == 200:
                scores = response.json()
                batch_time = end_time - start_time
                sentiment_times.append(batch_time)
                throughput = len(batch) / batch_time
                results['sentiment_results'].extend(scores)
                logger.info(f"   ✅ Sentiment: {len(batch)} articles in {batch_time:.2f}s ({throughput:.1f} art/sec)")
            else:
                error_msg = f"Sentiment batch {batch_idx + 1} failed: {response.status_code}"
                logger.error(f"   ❌ {error_msg}")
                results['errors'].append(error_msg)
                
        except Exception as e:
            error_msg = f"Sentiment batch {batch_idx + 1} error: {e}"
            logger.error(f"   ❌ {error_msg}")
            results['errors'].append(error_msg)
        
        # Test bias batch
        try:
            start_time = time.time()
            response = requests.post(f"{base_url}/score_bias_batch", json=payload, timeout=60)
            end_time = time.time()
            
            if response.status_code == 200:
                scores = response.json()
                batch_time = end_time - start_time
                bias_times.append(batch_time)
                throughput = len(batch) / batch_time
                results['bias_results'].extend(scores)
                logger.info(f"   ✅ Bias: {len(batch)} articles in {batch_time:.2f}s ({throughput:.1f} art/sec)")
            else:
                error_msg = f"Bias batch {batch_idx + 1} failed: {response.status_code}"
                logger.error(f"   ❌ {error_msg}")
                results['errors'].append(error_msg)
                
        except Exception as e:
            error_msg = f"Bias batch {batch_idx + 1} error: {e}"
            logger.error(f"   ❌ {error_msg}")
            results['errors'].append(error_msg)
    
    end_total = time.time()
    results['total_time'] = end_total - start_total
    
    # Calculate overall throughput
    if sentiment_times:
        total_sentiment_time = sum(sentiment_times)
        results['sentiment_throughput'] = len(articles) / total_sentiment_time
    
    if bias_times:
        total_bias_time = sum(bias_times)
        results['bias_throughput'] = len(articles) / total_bias_time
    
    return results

def print_performance_report(results: Dict):
    """Print comprehensive performance report"""
    print("\n" + "="*80)
    print("🏆 PRODUCTION STRESS TEST RESULTS")
    print("="*80)
    
    print(f"📊 Dataset: {results['total_articles']:,} articles")
    print(f"📦 Batch Size: {results['batch_size']}")
    print(f"⏱️  Total Test Time: {results['total_time']:.1f} seconds")
    
    if results['sentiment_results']:
        print(f"\n🎯 SENTIMENT ANALYSIS:")
        print(f"   Articles Processed: {len(results['sentiment_results']):,}")
        print(f"   Throughput: {results['sentiment_throughput']:.1f} articles/sec")
        print(f"   Average Score: {sum(results['sentiment_results'])/len(results['sentiment_results']):.3f}")
    
    if results['bias_results']:
        print(f"\n⚖️  BIAS ANALYSIS:")
        print(f"   Articles Processed: {len(results['bias_results']):,}")
        print(f"   Throughput: {results['bias_throughput']:.1f} articles/sec")
        print(f"   Average Score: {sum(results['bias_results'])/len(results['bias_results']):.3f}")
    
    if results['errors']:
        print(f"\n❌ ERRORS ({len(results['errors'])}):")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"   {error}")
        if len(results['errors']) > 5:
            print(f"   ... and {len(results['errors']) - 5} more errors")
    
    # Performance assessment
    print(f"\n🎉 PERFORMANCE ASSESSMENT:")
    if results['sentiment_throughput'] > 100:
        print("   ✅ Sentiment throughput: EXCELLENT")
    elif results['sentiment_throughput'] > 50:
        print("   ✅ Sentiment throughput: GOOD")  
    else:
        print("   ⚠️  Sentiment throughput: NEEDS IMPROVEMENT")
        
    if results['bias_throughput'] > 100:
        print("   ✅ Bias throughput: EXCELLENT")
    elif results['bias_throughput'] > 50:
        print("   ✅ Bias throughput: GOOD")
    else:
        print("   ⚠️  Bias throughput: NEEDS IMPROVEMENT")
    
    print("="*80)

def main():
    """Run production stress test"""
    print("🚀 Native TensorRT Production Stress Test")
    print("Testing with realistic article volumes and sizes")
    print("="*80)
    
    # Configuration
    NUM_ARTICLES = 1000
    AVG_ARTICLE_LENGTH = 2000
    BATCH_SIZE = 32
    BASE_URL = "http://localhost:8004"
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            logger.error("❌ Server not responding. Start the analyst agent first.")
            return 1
    except Exception as e:
        logger.error(f"❌ Cannot connect to server: {e}")
        logger.error("   Make sure the analyst agent is running on port 8004")
        return 1
    
    logger.info("✅ Server is responding")
    
    # Generate test dataset
    articles = generate_test_dataset(NUM_ARTICLES, AVG_ARTICLE_LENGTH)
    
    # Run stress test
    results = test_batch_performance(articles, BATCH_SIZE, BASE_URL)
    
    # Print results
    print_performance_report(results)
    
    # Determine success
    success = (
        len(results['sentiment_results']) == NUM_ARTICLES and
        len(results['bias_results']) == NUM_ARTICLES and
        len(results['errors']) == 0
    )
    
    if success:
        print("\n🎉 PRODUCTION STRESS TEST: PASSED")
        return 0
    else:
        print("\n⚠️  PRODUCTION STRESS TEST: ISSUES DETECTED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
