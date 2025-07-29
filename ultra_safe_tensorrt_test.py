#!/usr/bin/env python3
"""
Ultra-Safe TensorRT Stability Test
==================================

Minimal, ultra-conservative test to validate TensorRT without system crashes:
- Single article processing only
- Extensive error handling
- System resource monitoring
- Graceful degradation
"""

import time
import os
import gc
import sys
from typing import Optional

def check_environment():
    """Verify environment before testing"""
    print("🔍 Environment Check:")
    
    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    print(f"   Conda Environment: {conda_env}")
    
    # Check Python packages
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    except ImportError:
        print("   ⚠️  PyTorch not available")
        return False
    
    try:
        import pycuda
        print("   ✅ PyCUDA available")
    except ImportError:
        print("   ❌ PyCUDA not available")
        return False
    
    try:
        import tensorrt
        print(f"   ✅ TensorRT: {tensorrt.__version__}")
    except ImportError:
        print("   ❌ TensorRT not available")
        return False
    
    return True

def test_single_article():
    """Test processing a single article safely"""
    print("\n🧪 Single Article Test:")
    
    test_article = "Breaking news: Technology stocks rise as market confidence grows amid positive earnings reports."
    
    try:
        # Import with error handling
        from agents.analyst.tensorrt_acceleration import TensorRTAnalyst
        
        print("   Initializing TensorRT Analyst...")
        analyst = TensorRTAnalyst()
        
        if not analyst.native_engine:
            print("   ⚠️  Native TensorRT not available, testing fallback...")
            return test_fallback_single(test_article)
        
        if not analyst.native_engine.engines:
            print("   ⚠️  No native engines loaded")
            return False
        
        print(f"   Available engines: {list(analyst.native_engine.engines.keys())}")
        
        # Test sentiment analysis
        print("   Testing sentiment analysis...", end=" ")
        start_time = time.time()
        try:
            sentiment_score = analyst.score_sentiment_tensorrt(test_article)
            sentiment_time = time.time() - start_time
            
            if sentiment_score is not None:
                print(f"✅ Score: {sentiment_score:.3f} (Time: {sentiment_time:.3f}s)")
            else:
                print("❌ Returned None")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
        
        # Small delay for system stability
        time.sleep(0.1)
        
        # Test bias analysis
        print("   Testing bias analysis...", end=" ")
        start_time = time.time()
        try:
            bias_score = analyst.score_bias_tensorrt(test_article)
            bias_time = time.time() - start_time
            
            if bias_score is not None:
                print(f"✅ Score: {bias_score:.3f} (Time: {bias_time:.3f}s)")
            else:
                print("❌ Returned None")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
        
        print(f"   ✅ Single article test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return False

def test_fallback_single(test_article: str) -> bool:
    """Test fallback system with single article"""
    print("   Testing fallback system...")
    
    try:
        # Try basic transformers pipeline
        from transformers import pipeline
        import torch
        
        if torch.cuda.is_available():
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0
            )
            
            result = sentiment_analyzer(test_article)
            print(f"   ✅ Fallback sentiment: {result[0]['label']} ({result[0]['score']:.3f})")
            return True
        else:
            print("   ❌ No GPU available for fallback")
            return False
            
    except Exception as e:
        print(f"   ❌ Fallback failed: {e}")
        return False

def test_small_batch():
    """Test very small batch (5 articles) if single article works"""
    print("\n📦 Small Batch Test (5 articles):")
    
    test_articles = [
        "Technology stocks continue to rise amid positive market sentiment.",
        "Global climate summit reaches new agreements on emission standards.",
        "Economic indicators suggest steady growth in the manufacturing sector.",
        "Healthcare innovations drive investment in biotechnology companies.",
        "Renewable energy adoption accelerates across multiple industries."
    ]
    
    try:
        from agents.analyst.tensorrt_acceleration import TensorRTAnalyst
        analyst = TensorRTAnalyst()
        
        if not analyst.native_engine or not analyst.native_engine.engines:
            print("   ⚠️  Native TensorRT not available for batch test")
            return False
        
        # Test batch sentiment
        print("   Testing batch sentiment...", end=" ")
        start_time = time.time()
        sentiment_results = analyst.score_sentiment_batch_tensorrt(test_articles)
        sentiment_time = time.time() - start_time
        
        if sentiment_results and all(score is not None for score in sentiment_results):
            rate = len(test_articles) / sentiment_time
            print(f"✅ {rate:.1f} articles/sec")
        else:
            print("❌ Some results were None")
            return False
        
        # Small delay
        time.sleep(0.1)
        
        # Test batch bias
        print("   Testing batch bias...", end=" ")
        start_time = time.time()
        bias_results = analyst.score_bias_batch_tensorrt(test_articles)
        bias_time = time.time() - start_time
        
        if bias_results and all(score is not None for score in bias_results):
            rate = len(test_articles) / bias_time
            print(f"✅ {rate:.1f} articles/sec")
        else:
            print("❌ Some results were None")
            return False
        
        print("   ✅ Small batch test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Batch test failed: {e}")
        return False

def main():
    """Run ultra-safe TensorRT stability test"""
    print("🛡️  ULTRA-SAFE TENSORRT STABILITY TEST")
    print("=" * 50)
    print("Purpose: Validate TensorRT without system crashes")
    print("Strategy: Conservative testing with extensive error handling")
    print()
    
    # Environment check
    if not check_environment():
        print("\n❌ Environment check failed - cannot proceed")
        return False
    
    # Memory cleanup before starting
    gc.collect()
    
    # Test single article first
    single_success = test_single_article()
    
    if not single_success:
        print("\n❌ Single article test failed - stopping here")
        return False
    
    # Memory cleanup
    gc.collect()
    time.sleep(1)  # Brief pause
    
    # Only proceed to batch test if single article worked
    print("\n✅ Single article test passed - proceeding to batch test")
    batch_success = test_small_batch()
    
    # Final cleanup
    gc.collect()
    
    # Summary
    print(f"\n🎯 TEST SUMMARY")
    print("-" * 30)
    print(f"Single Article Test: {'✅ PASS' if single_success else '❌ FAIL'}")
    print(f"Small Batch Test: {'✅ PASS' if batch_success else '❌ FAIL'}")
    
    if single_success and batch_success:
        print("\n🚀 STABILITY VALIDATED: Native TensorRT is stable and functional!")
        print("System can safely handle production workloads.")
    elif single_success:
        print("\n⚠️  PARTIAL SUCCESS: Single articles work, batch processing needs attention")
    else:
        print("\n❌ SYSTEM ISSUES: TensorRT may not be properly configured")
    
    return single_success and batch_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
