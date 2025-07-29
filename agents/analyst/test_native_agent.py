#!/usr/bin/env python3
"""
Test script for the Native TensorRT Analyst Agent
Validates the updated FastAPI endpoints with native TensorRT implementation
"""

import sys
import os
import logging
import time
import requests
import json
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_agent_health(base_url: str = "http://localhost:8004") -> bool:
    """Test if the agent is healthy and responding"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Agent health check passed")
            return True
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False

def test_sentiment_scoring(base_url: str = "http://localhost:8004") -> bool:
    """Test individual sentiment scoring"""
    test_text = "This is a fantastic breakthrough in renewable energy technology!"
    
    try:
        payload = {
            "args": [test_text],
            "kwargs": {}
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/score_sentiment", json=payload, timeout=10)
        end_time = time.time()
        
        if response.status_code == 200:
            score = response.json()
            logger.info(f"✅ Sentiment scoring: {score:.3f} ({end_time-start_time:.3f}s)")
            return True
        else:
            logger.error(f"❌ Sentiment scoring failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Sentiment scoring error: {e}")
        return False

def test_bias_scoring(base_url: str = "http://localhost:8004") -> bool:
    """Test individual bias scoring"""
    test_text = "The government's new policy has been implemented across all states."
    
    try:
        payload = {
            "args": [test_text],
            "kwargs": {}
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/score_bias", json=payload, timeout=10)
        end_time = time.time()
        
        if response.status_code == 200:
            score = response.json()
            logger.info(f"✅ Bias scoring: {score:.3f} ({end_time-start_time:.3f}s)")
            return True
        else:
            logger.error(f"❌ Bias scoring failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Bias scoring error: {e}")
        return False

def test_batch_processing(base_url: str = "http://localhost:8004") -> bool:
    """Test native TensorRT batch processing"""
    test_texts = [
        "Breaking news: Stock market reaches record highs as investors show confidence.",
        "Local community celebrates opening of new sustainable energy facility.",
        "Weather forecast predicts severe storms approaching coastal regions this weekend.",
        "Scientists announce breakthrough in renewable energy storage technology.",
        "City council approves budget for infrastructure improvements and expansion."
    ]
    
    try:
        # Test sentiment batch
        payload = {
            "args": [test_texts],
            "kwargs": {}
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/score_sentiment_batch", json=payload, timeout=15)
        end_time = time.time()
        
        if response.status_code == 200:
            scores = response.json()
            batch_time = end_time - start_time
            throughput = len(test_texts) / batch_time
            logger.info(f"✅ Sentiment batch: {len(scores)} articles ({throughput:.1f} art/sec)")
        else:
            logger.error(f"❌ Sentiment batch failed: {response.status_code}")
            return False
        
        # Test bias batch
        start_time = time.time()
        response = requests.post(f"{base_url}/score_bias_batch", json=payload, timeout=15)
        end_time = time.time()
        
        if response.status_code == 200:
            scores = response.json()
            batch_time = end_time - start_time
            throughput = len(test_texts) / batch_time
            logger.info(f"✅ Bias batch: {len(scores)} articles ({throughput:.1f} art/sec)")
            return True
        else:
            logger.error(f"❌ Bias batch failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Batch processing error: {e}")
        return False

def test_analyze_article(base_url: str = "http://localhost:8004") -> bool:
    """Test the combined article analysis endpoint"""
    test_text = "Major technology companies announced a collaborative effort to develop sustainable computing solutions that could reduce global energy consumption. The initiative brings together industry leaders and environmental experts to create next-generation hardware optimized for efficiency."
    
    try:
        payload = {
            "args": [test_text],
            "kwargs": {}
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/analyze_article", json=payload, timeout=10)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            processing_time = end_time - start_time
            logger.info(f"✅ Article analysis: sentiment={result['sentiment']:.3f}, bias={result['bias']:.3f} ({processing_time:.3f}s)")
            return True
        else:
            logger.error(f"❌ Article analysis failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Article analysis error: {e}")
        return False

def test_engine_info(base_url: str = "http://localhost:8004") -> bool:
    """Test the engine information endpoint"""
    try:
        response = requests.get(f"{base_url}/engine_info", timeout=5)
        
        if response.status_code == 200:
            info = response.json()
            logger.info(f"✅ Engine info retrieved: {info.get('status', 'unknown')}")
            if 'engines' in info:
                for engine_name, engine_data in info['engines'].items():
                    logger.info(f"   {engine_name}: loaded={engine_data.get('loaded', False)}")
            return True
        else:
            logger.error(f"❌ Engine info failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Engine info error: {e}")
        return False

def main():
    """Run all tests for the Native TensorRT Analyst Agent"""
    print("🚀 Testing Native TensorRT Analyst Agent")
    print("=" * 50)
    
    base_url = "http://localhost:8004"
    
    # Test sequence
    tests = [
        ("Health Check", test_agent_health),
        ("Engine Information", test_engine_info),
        ("Sentiment Scoring", test_sentiment_scoring),
        ("Bias Scoring", test_bias_scoring),
        ("Batch Processing", test_batch_processing),
        ("Article Analysis", test_analyze_article),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name}...")
        try:
            if test_func(base_url):
                passed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Native TensorRT Agent is operational!")
        return 0
    else:
        print("⚠️ Some tests failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
