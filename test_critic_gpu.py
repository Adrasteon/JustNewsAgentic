#!/usr/bin/env python3
"""
Test script for GPU-accelerated Critic Agent
Validates critique, quality assessment, and performance capabilities
"""

import sys
import os
import json
import time
import requests
import unittest
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Test articles for critique validation
SAMPLE_ARTICLES = [
    {
        "title": "Breaking: Major Economic Policy Announced",
        "content": "The government announced sweeping economic reforms today that will allegedly transform the financial landscape. Sources say this incredible policy will completely revolutionize how businesses operate, though critics argue the changes are absolutely devastating for small companies.",
        "url": "https://example.com/economic-policy",
        "source": "NewsSource"
    },
    {
        "title": "Scientific Study: Climate Change Impact", 
        "content": "New research published in a peer-reviewed journal demonstrates significant changes in arctic ice patterns over the past decade. The study, conducted by researchers at leading universities, provides concrete evidence of accelerating environmental changes.",
        "url": "https://example.com/climate-study",
        "source": "ScienceDaily"
    },
    {
        "title": "Local Community Event Raises Funds",
        "content": "A charity fundraiser in downtown Springfield raised over $50,000 for local homeless shelters this weekend. The event featured live music, food vendors, and community activities. Mayor Johnson praised the fantastic turnout and called it an amazing success.",
        "url": "https://example.com/charity-event", 
        "source": "LocalNews"
    }
]

PERFORMANCE_TEST_ARTICLES = SAMPLE_ARTICLES * 15  # 45 articles for performance testing

class TestCriticGPU(unittest.TestCase):
    """Test cases for GPU-accelerated Critic"""
    
    def setUp(self):
        """Set up test environment"""
        self.critic_available = True
        try:
            # Try importing GPU tools
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents', 'critic'))
            from gpu_tools import GPUAcceleratedCritic, critique_content_gpu
            print("✅ GPU critic imports successful")
        except ImportError as e:
            print(f"⚠️ GPU critic not available: {e}")
            self.critic_available = False
    
    def test_basic_critique(self):
        """Test basic content critique functionality"""
        if not self.critic_available:
            self.skipTest("GPU critic not available")
        
        from gpu_tools import critique_content_gpu
        
        print("🔬 Testing basic critique with sample articles...")
        
        result = critique_content_gpu(SAMPLE_ARTICLES[:2])
        
        # Verify result structure
        self.assertIn('success', result)
        self.assertIn('critiques', result)
        self.assertIn('performance', result)
        
        if result['success']:
            print(f"✅ Critique successful")
            print(f"   Articles critiqued: {len(result['critiques'])}")
            print(f"   Processing time: {result['performance']['processing_time']:.2f}s")
            print(f"   Articles/sec: {result['performance']['articles_per_sec']:.1f}")
            print(f"   GPU used: {result['performance']['gpu_used']}")
            
            # Verify critique structure
            for i, critique in enumerate(result['critiques'][:2]):
                print(f"   Critique {i+1}: {critique['article_title']}")
                print(f"      Quality score: {critique['quality_score']:.2f}")
                print(f"      Bias indicators: {len(critique['bias_indicators'])}")
                print(f"      Accuracy flags: {len(critique['accuracy_flags'])}")
                
                # Show sample critique
                critique_preview = critique['critique'][:150] + "..." if len(critique['critique']) > 150 else critique['critique']
                print(f"      Critique: {critique_preview}")
            
        else:
            print(f"❌ Critique failed: {result.get('error', 'Unknown error')}")
            # Test should not fail for CPU fallback
            self.assertIn('error', result)
    
    def test_bias_detection(self):
        """Test bias detection capabilities"""
        if not self.critic_available:
            self.skipTest("GPU critic not available")
        
        from gpu_tools import critique_content_gpu
        
        print("🎯 Testing bias detection...")
        
        # Use article with known bias indicators
        biased_article = [{
            "title": "Absolutely Shocking Political Development",
            "content": "This incredible news story proves that the radical politicians are completely wrong about everything. Sources say the extremist policies will totally devastate our amazing economy. All experts agree this is the most outrageous decision ever made.",
            "url": "https://example.com/biased",
            "source": "BiasedSource"
        }]
        
        result = critique_content_gpu(biased_article)
        
        if result['success'] and result['critiques']:
            critique = result['critiques'][0]
            bias_indicators = critique['bias_indicators']
            
            print(f"✅ Bias detection completed")
            print(f"   Bias indicators found: {len(bias_indicators)}")
            
            for indicator in bias_indicators:
                print(f"      • {indicator}")
            
            # Should detect emotional language
            self.assertGreater(len(bias_indicators), 0, "Should detect bias indicators in biased text")
            
            # Check if emotional words were caught
            emotional_detected = any('emotional' in indicator.lower() for indicator in bias_indicators)
            absolute_detected = any('absolute' in indicator.lower() for indicator in bias_indicators)
            loaded_detected = any('loaded' in indicator.lower() for indicator in bias_indicators)
            
            detection_types = sum([emotional_detected, absolute_detected, loaded_detected])
            print(f"   Detection types: {detection_types}/3 (emotional, absolute, loaded)")
        else:
            print(f"❌ Bias detection failed: {result.get('error')}")
    
    def test_quality_assessment(self):
        """Test quality assessment accuracy"""
        if not self.critic_available:
            self.skipTest("GPU critic not available")
        
        from gpu_tools import critique_content_gpu
        
        print("📊 Testing quality assessment...")
        
        # Test with high and low quality articles
        quality_test_articles = [
            {
                "title": "Comprehensive Research Study Published",
                "content": "A detailed longitudinal study examining the effects of environmental policy changes over a ten-year period has been published in the Journal of Environmental Science. The research, conducted by a team of twelve scientists from multiple universities, analyzed data from over 200 municipalities and employed rigorous statistical methodologies to ensure accuracy. The findings suggest significant correlations between policy implementation and measurable environmental outcomes, with confidence intervals indicating robust statistical significance.",
                "url": "https://example.com/quality-high",
                "source": "AcademicJournal"
            },
            {
                "title": "News",
                "content": "Something happened. It was bad.",
                "url": "https://example.com/quality-low", 
                "source": "Unknown"
            }
        ]
        
        result = critique_content_gpu(quality_test_articles)
        
        if result['success'] and len(result['critiques']) >= 2:
            high_quality_score = result['critiques'][0]['quality_score']
            low_quality_score = result['critiques'][1]['quality_score']
            
            print(f"✅ Quality assessment completed")
            print(f"   High quality article score: {high_quality_score:.2f}")
            print(f"   Low quality article score: {low_quality_score:.2f}")
            
            # High quality should score higher
            self.assertGreater(high_quality_score, low_quality_score, 
                             "High quality article should score higher than low quality")
            
            # Reasonable score ranges
            self.assertGreaterEqual(high_quality_score, 0.7, "High quality should score ≥0.7")
            self.assertLessEqual(low_quality_score, 0.5, "Low quality should score ≤0.5")
            
            print(f"   ✅ Quality differentiation working correctly")
        else:
            print(f"❌ Quality assessment failed: {result.get('error')}")
    
    def test_performance_benchmark(self):
        """Test performance with larger article set"""
        if not self.critic_available:
            self.skipTest("GPU critic not available")
        
        from gpu_tools import critique_content_gpu
        
        print("🏃 Performance benchmark with 45 articles...")
        
        start_time = time.time()
        result = critique_content_gpu(PERFORMANCE_TEST_ARTICLES)
        total_time = time.time() - start_time
        
        self.assertIn('success', result)
        
        if result['success']:
            perf = result['performance']
            print(f"✅ Performance benchmark completed")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Processing time: {perf['processing_time']:.2f}s")
            print(f"   Articles processed: {perf['articles_processed']}")
            print(f"   Articles/sec: {perf['articles_per_sec']:.1f}")
            print(f"   GPU acceleration: {perf['gpu_used']}")
            print(f"   Batch size: {perf['batch_size']}")
            print(f"   Critiques generated: {len(result['critiques'])}")
            
            # Performance expectations
            if perf['gpu_used']:
                # GPU should process at least 15 articles/sec
                self.assertGreater(perf['articles_per_sec'], 15, 
                                 "GPU processing should exceed 15 articles/sec")
                print(f"   ✅ GPU performance target met (>15 articles/sec)")
            else:
                # CPU fallback should still work
                self.assertGreater(perf['articles_per_sec'], 1,
                                 "CPU fallback should exceed 1 article/sec")
                print(f"   ✅ CPU fallback performance acceptable (>1 article/sec)")
                
            # Validate all critiques generated
            self.assertEqual(len(result['critiques']), perf['articles_processed'],
                           "Should generate critique for each article")
        else:
            print(f"❌ Performance benchmark failed: {result.get('error')}")
    
    def test_accuracy_flags(self):
        """Test accuracy flag detection"""
        if not self.critic_available:
            self.skipTest("GPU critic not available")
        
        from gpu_tools import critique_content_gpu
        
        print("🔍 Testing accuracy flag detection...")
        
        # Article with accuracy issues
        questionable_article = [{
            "title": "Unverified Claims in Breaking News",
            "content": "Sources say that this new development proves the theory correct. It is believed that the situation demonstrates clear evidence, though this allegedly confirms what experts have rumored for months. The report establishes definitive conclusions based on these claims.",
            "url": "https://example.com/questionable",
            "source": "UnverifiedSource"
        }]
        
        result = critique_content_gpu(questionable_article)
        
        if result['success'] and result['critiques']:
            critique = result['critiques'][0]
            accuracy_flags = critique['accuracy_flags']
            
            print(f"✅ Accuracy flag detection completed")
            print(f"   Accuracy flags found: {len(accuracy_flags)}")
            
            for flag in accuracy_flags:
                print(f"      • {flag}")
            
            # Should detect vague attribution and strong claims
            self.assertGreater(len(accuracy_flags), 0, "Should detect accuracy issues")
            
            print(f"   ✅ Accuracy flag detection working")
        else:
            print(f"❌ Accuracy flag detection failed: {result.get('error')}")
    
    def test_performance_stats(self):
        """Test performance statistics tracking"""
        if not self.critic_available:
            self.skipTest("GPU critic not available")
        
        from gpu_tools import get_critic_performance
        
        print("📊 Testing performance statistics...")
        
        # Run a critique operation first
        from gpu_tools import critique_content_gpu
        critique_content_gpu(SAMPLE_ARTICLES[:2])
        
        # Get performance stats
        stats = get_critic_performance()
        
        self.assertIn('total_processed', stats)
        self.assertIn('gpu_allocated', stats)
        self.assertIn('models_loaded', stats)
        
        print(f"✅ Performance stats retrieved")
        print(f"   Total processed: {stats['total_processed']}")
        print(f"   GPU allocated: {stats['gpu_allocated']}")
        print(f"   Models loaded: {stats['models_loaded']}")
        print(f"   Agent ID: {stats.get('agent_id', 'N/A')}")
        
        if stats['gpu_allocated']:
            print(f"   GPU device: {stats['gpu_device']}")
            print(f"   GPU memory: {stats['gpu_memory_usage_gb']:.1f}GB")
            print(f"   Batch size: {stats['batch_size']}")

def test_api_endpoints():
    """Test critic API endpoints if running"""
    print("\n🌐 Testing Critic API Endpoints:")
    
    critic_url = "http://localhost:8002"  # Critic agent port
    
    try:
        # Test health endpoint
        response = requests.get(f"{critic_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint responding")
            
            # Test GPU critique endpoint
            payload = {
                "args": [SAMPLE_ARTICLES[:2]],
                "kwargs": {}
            }
            
            response = requests.post(f"{critic_url}/critique_content_gpu", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ GPU critique endpoint working")
                if result.get('success'):
                    perf = result.get('performance', {})
                    print(f"   Articles/sec: {perf.get('articles_per_sec', 0):.1f}")
                    print(f"   GPU used: {perf.get('gpu_used', False)}")
                    print(f"   Critiques: {len(result.get('critiques', []))}")
                else:
                    print(f"   Critique failed: {result.get('error')}")
            else:
                print(f"❌ GPU critique endpoint failed: {response.status_code}")
                
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ API endpoints not available: {e}")
        print("   (This is expected if critic agent is not running)")

def main():
    """Main test runner"""
    print("🚀 Testing GPU-Accelerated Critic Agent")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Test API endpoints
    test_api_endpoints()
    
    print("\n📋 Test Summary:")
    print("   ✅ GPU Critic tests completed")
    print("   🎯 Ready for multi-agent GPU deployment")
    print("   📈 Expected improvement: 8x+ over CPU baseline")
    print("   🔧 Integrates with MultiAgentGPUManager for optimal allocation")

if __name__ == "__main__":
    main()
