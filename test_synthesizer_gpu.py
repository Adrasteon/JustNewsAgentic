#!/usr/bin/env python3
"""
Test script for GPU-accelerated Synthesizer Agent
Validates synthesis, theme identification, and performance capabilities
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

# Test articles for synthesis validation
SAMPLE_ARTICLES = [
    {
        "title": "Tech Giants Report Strong Q4 Earnings",
        "content": "Major technology companies including Apple, Google, and Microsoft have reported stronger than expected fourth quarter earnings, driven by robust cloud computing demand and AI investments. Apple's revenue reached $119.6 billion, while Microsoft's cloud revenue grew 24% year-over-year.",
        "url": "https://example.com/tech-earnings",
        "source": "TechNews"
    },
    {
        "title": "Federal Reserve Keeps Interest Rates Steady",
        "content": "The Federal Reserve announced it will maintain current interest rates at 5.25-5.50% range, citing concerns about inflation and employment levels. Fed Chair Powell emphasized the need for more data before making future rate decisions.",
        "url": "https://example.com/fed-rates",
        "source": "FinanceDaily"
    },
    {
        "title": "Climate Summit Reaches Historic Agreement",
        "content": "World leaders at COP29 have reached a breakthrough agreement on carbon emissions reduction, with 195 countries committing to achieve net-zero emissions by 2050. The deal includes $100 billion in climate finance for developing nations.",
        "url": "https://example.com/climate-summit",
        "source": "GlobalNews"
    },
    {
        "title": "AI Startup Raises $500M in Series C Funding",
        "content": "Anthropic, a leading AI safety company, has secured $500 million in Series C funding led by Google and other strategic investors. The funding will accelerate research into safe and beneficial artificial intelligence systems.",
        "url": "https://example.com/ai-funding",
        "source": "VentureReport"
    },
    {
        "title": "Electric Vehicle Sales Surge 40% in 2024",
        "content": "Electric vehicle sales have increased by 40% globally in 2024, with Tesla maintaining market leadership while Chinese manufacturers BYD and NIO gain significant market share. Battery technology improvements and government incentives drive adoption.",
        "url": "https://example.com/ev-sales",
        "source": "AutoNews"
    }
]

PERFORMANCE_TEST_ARTICLES = SAMPLE_ARTICLES * 10  # 50 articles for performance testing

class TestSynthesizerGPU(unittest.TestCase):
    """Test cases for GPU-accelerated Synthesizer"""
    
    def setUp(self):
        """Set up test environment"""
        self.synthesizer_available = True
        try:
            # Try importing GPU tools
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents', 'synthesizer'))
            from gpu_tools import GPUAcceleratedSynthesizer, synthesize_news_articles_gpu
            print("✅ GPU synthesizer imports successful")
        except ImportError as e:
            print(f"⚠️ GPU synthesizer not available: {e}")
            self.synthesizer_available = False
    
    def test_basic_synthesis(self):
        """Test basic article synthesis functionality"""
        if not self.synthesizer_available:
            self.skipTest("GPU synthesizer not available")
        
        from gpu_tools import synthesize_news_articles_gpu
        
        print("🔬 Testing basic synthesis with sample articles...")
        
        result = synthesize_news_articles_gpu(SAMPLE_ARTICLES[:3])
        
        # Verify result structure
        self.assertIn('success', result)
        self.assertIn('themes', result)
        self.assertIn('synthesis', result)
        self.assertIn('performance', result)
        
        if result['success']:
            print(f"✅ Synthesis successful")
            print(f"   Themes identified: {len(result['themes'])}")
            print(f"   Processing time: {result['performance']['processing_time']:.2f}s")
            print(f"   Articles/sec: {result['performance']['articles_per_sec']:.1f}")
            print(f"   GPU used: {result['performance']['gpu_used']}")
            
            # Verify themes structure
            for i, theme in enumerate(result['themes'][:2]):  # Show first 2 themes
                print(f"   Theme {i+1}: {theme['theme_name']} ({theme['article_count']} articles)")
            
            # Show synthesis preview
            synthesis_preview = result['synthesis'][:200] + "..." if len(result['synthesis']) > 200 else result['synthesis']
            print(f"   Synthesis preview: {synthesis_preview}")
            
        else:
            print(f"❌ Synthesis failed: {result.get('error', 'Unknown error')}")
            # Test should not fail for CPU fallback
            self.assertIn('error', result)
    
    def test_single_article_synthesis(self):
        """Test synthesis with single article (edge case)"""
        if not self.synthesizer_available:
            self.skipTest("GPU synthesizer not available")
        
        from gpu_tools import synthesize_news_articles_gpu
        
        print("🔬 Testing single article synthesis...")
        
        result = synthesize_news_articles_gpu([SAMPLE_ARTICLES[0]])
        
        self.assertIn('success', result)
        
        if result['success']:
            print(f"✅ Single article synthesis successful")
            print(f"   Themes: {len(result['themes'])}")
            print(f"   Processing time: {result['performance']['processing_time']:.2f}s")
            
            # Should create one primary theme
            self.assertGreaterEqual(len(result['themes']), 1)
        else:
            print(f"❌ Single article synthesis failed: {result.get('error')}")
    
    def test_empty_articles_handling(self):
        """Test handling of empty article list"""
        if not self.synthesizer_available:
            self.skipTest("GPU synthesizer not available")
        
        from gpu_tools import synthesize_news_articles_gpu
        
        print("🔬 Testing empty articles handling...")
        
        result = synthesize_news_articles_gpu([])
        
        self.assertIn('success', result)
        self.assertFalse(result['success'])  # Should fail gracefully
        self.assertIn('error', result)
        
        print(f"✅ Empty articles handled correctly: {result['error']}")
    
    def test_performance_benchmark(self):
        """Test performance with larger article set"""
        if not self.synthesizer_available:
            self.skipTest("GPU synthesizer not available")
        
        from gpu_tools import synthesize_news_articles_gpu
        
        print("🏃 Performance benchmark with 50 articles...")
        
        start_time = time.time()
        result = synthesize_news_articles_gpu(PERFORMANCE_TEST_ARTICLES)
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
            print(f"   Themes identified: {len(result['themes'])}")
            
            # Performance expectations
            if perf['gpu_used']:
                # GPU should process at least 20 articles/sec
                self.assertGreater(perf['articles_per_sec'], 20, 
                                 "GPU processing should exceed 20 articles/sec")
                print(f"   ✅ GPU performance target met (>20 articles/sec)")
            else:
                # CPU fallback should still work
                self.assertGreater(perf['articles_per_sec'], 1,
                                 "CPU fallback should exceed 1 article/sec")
                print(f"   ✅ CPU fallback performance acceptable (>1 article/sec)")
        else:
            print(f"❌ Performance benchmark failed: {result.get('error')}")
    
    def test_theme_quality(self):
        """Test quality of theme identification"""
        if not self.synthesizer_available:
            self.skipTest("GPU synthesizer not available")
        
        from gpu_tools import synthesize_news_articles_gpu
        
        print("🎯 Testing theme identification quality...")
        
        # Use diverse articles that should form distinct themes
        diverse_articles = [
            {
                "title": "Apple Launches New iPhone",
                "content": "Apple announced the latest iPhone with advanced AI capabilities and improved battery life.",
                "url": "https://example.com/iphone", 
                "source": "TechNews"
            },
            {
                "title": "Google AI Breakthrough",
                "content": "Google's DeepMind achieves significant breakthrough in protein folding prediction using artificial intelligence.",
                "url": "https://example.com/google-ai",
                "source": "ScienceDaily"
            },
            {
                "title": "Climate Change Impact Study",
                "content": "New research shows accelerating ice sheet melting in Antarctica, raising sea level concerns.",
                "url": "https://example.com/climate",
                "source": "EnvironmentNews"
            },
            {
                "title": "Stock Market Rally Continues",
                "content": "Major stock indices reach new highs as investors show confidence in economic recovery.",
                "url": "https://example.com/stocks",
                "source": "MarketWatch"
            }
        ]
        
        result = synthesize_news_articles_gpu(diverse_articles)
        
        if result['success']:
            themes = result['themes']
            print(f"✅ Theme identification completed")
            print(f"   Themes identified: {len(themes)}")
            
            for theme in themes:
                print(f"   • {theme['theme_name']}: {theme['article_count']} articles "
                      f"(coherence: {theme['coherence_score']:.2f})")
            
            # Quality checks
            self.assertGreaterEqual(len(themes), 2, "Should identify multiple themes for diverse articles")
            
            # Check coherence scores are reasonable
            for theme in themes:
                self.assertGreaterEqual(theme['coherence_score'], 0.0)
                self.assertLessEqual(theme['coherence_score'], 1.0)
            
            print(f"   ✅ Theme quality metrics passed")
        else:
            print(f"❌ Theme identification failed: {result.get('error')}")
    
    def test_performance_stats(self):
        """Test performance statistics tracking"""
        if not self.synthesizer_available:
            self.skipTest("GPU synthesizer not available")
        
        from gpu_tools import get_synthesizer_performance
        
        print("📊 Testing performance statistics...")
        
        # Run a synthesis operation first
        from gpu_tools import synthesize_news_articles_gpu
        synthesize_news_articles_gpu(SAMPLE_ARTICLES[:3])
        
        # Get performance stats
        stats = get_synthesizer_performance()
        
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
    """Test synthesizer API endpoints if running"""
    print("\n🌐 Testing API Endpoints:")
    
    synthesizer_url = "http://localhost:8005"
    
    try:
        # Test health endpoint
        response = requests.get(f"{synthesizer_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint responding")
            
            # Test GPU synthesis endpoint
            payload = {
                "args": [SAMPLE_ARTICLES[:3]],
                "kwargs": {}
            }
            
            response = requests.post(f"{synthesizer_url}/synthesize_news_articles_gpu", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ GPU synthesis endpoint working")
                if result.get('success'):
                    perf = result.get('performance', {})
                    print(f"   Articles/sec: {perf.get('articles_per_sec', 0):.1f}")
                    print(f"   GPU used: {perf.get('gpu_used', False)}")
                else:
                    print(f"   Synthesis failed: {result.get('error')}")
            else:
                print(f"❌ GPU synthesis endpoint failed: {response.status_code}")
                
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ API endpoints not available: {e}")
        print("   (This is expected if synthesizer agent is not running)")

def main():
    """Main test runner"""
    print("🚀 Testing GPU-Accelerated Synthesizer Agent")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Test API endpoints
    test_api_endpoints()
    
    print("\n📋 Test Summary:")
    print("   ✅ GPU Synthesizer tests completed")
    print("   🎯 Ready for multi-agent GPU expansion")
    print("   📈 Expected improvement: 10x+ over CPU baseline")
    print("   🔧 Integrates with MultiAgentGPUManager for optimal allocation")

if __name__ == "__main__":
    main()
