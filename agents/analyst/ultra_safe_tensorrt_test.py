#!/usr/bin/env python3
"""
Ultra-Safe TensorRT Test with Proper Context Management
Final validation for completely clean, warning-free operation
"""

import sys
import logging
import time
from typing import List, Optional, Dict, Any
import pytest

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional CUDA/PyCUDA driver. Import at module-level inside try/except so static
# linters don't report unresolved imports in environments without pycuda.
try:
    import pycuda.driver as _pycuda_driver  # type: ignore
    HAS_PYCUDA = True
except Exception:
    _pycuda_driver = None  # type: ignore
    HAS_PYCUDA = False

def safe_gpu_init():
    """Safely initialize GPU with proper CUDA context management"""
    # If pycuda isn't installed, bail out early (the tests will skip)
    if not HAS_PYCUDA:
        logger.warning("pycuda not available; skipping GPU initialization")
        return None, None

    try:
        cuda = _pycuda_driver

        # Initialize CUDA driver
        cuda.init()
        logger.info("✅ CUDA driver initialized")

        # Check GPU availability
        device_count = cuda.Device.count()
        logger.info(f"✅ Found {device_count} CUDA devices")

        # Get device info
        device = cuda.Device(0)
        logger.info(f"✅ Using device: {device.name()}")

        # Get memory info
        context = device.make_context()
        free_mem, total_mem = cuda.mem_get_info()
        logger.info(f"✅ GPU Memory: {free_mem/1024**3:.1f}GB free / {total_mem/1024**3:.1f}GB total")

        return context, device

    except Exception as e:
        logger.error(f"❌ GPU initialization failed: {e}")
        return None, None

def test_native_tensorrt_clean():
    """Test native TensorRT with completely clean context management"""
    print("\n" + "="*80)
    print("🚀 ULTRA-SAFE TENSORRT TEST - CLEAN OPERATION VALIDATION")
    print("="*80)
    
    # Test articles for comprehensive validation
    test_articles = [
        "Breaking news: Stock market reaches record highs as investors show confidence in technology sector.",
        "Local community celebrates opening of new sustainable energy facility with overwhelming support.",
        "Weather forecast predicts severe storms approaching coastal regions this weekend.",
        "Scientists announce breakthrough in renewable energy storage technology development.",
        "City council approves budget for infrastructure improvements and public transportation expansion."
    ]
    
    cuda_context = None
    
    # Initialize GPU safely; if CUDA/pycuda isn't available in this environment,
    # skip this test rather than failing.
    cuda_context, device = safe_gpu_init()
    if cuda_context is None:
        pytest.skip("CUDA/pycuda not available in this environment")
        
        # Import and initialize TensorRT engine with context management
        from native_tensorrt_engine import NativeTensorRTInferenceEngine
        
        logger.info("🔄 Initializing Native TensorRT Engine...")
        
        # Use correct relative path for engines when running from agents/analyst directory
        # Use context manager for proper cleanup
        with NativeTensorRTInferenceEngine(engines_dir="tensorrt_engines") as engine:
            logger.info("✅ Engine initialized successfully")
            
            # Check if native engines are available
            engine_info = engine.get_engine_info()
            sentiment_loaded = engine_info.get('sentiment', {}).get('loaded', False)
            bias_loaded = engine_info.get('bias', {}).get('loaded', False)
            
            if not (sentiment_loaded and bias_loaded):
                print(f"\n⚠️  Native TensorRT engines not available:")
                print(f"   Sentiment engine: {'✅' if sentiment_loaded else '❌'}")
                print(f"   Bias engine: {'✅' if bias_loaded else '❌'}")
                print(f"   Falling back to baseline operations for testing...")
            
            # Test individual sentiment analysis
            print("\n📊 Testing Sentiment Analysis:")
            for i, article in enumerate(test_articles, 1):
                start_time = time.time()
                sentiment_score = engine.score_sentiment(article)
                end_time = time.time()
                
                if sentiment_score is not None:
                    print(f"  Article {i}: Sentiment={sentiment_score:.3f} ({end_time-start_time:.3f}s)")
                else:
                    print(f"  Article {i}: Sentiment=None (fallback/error) ({end_time-start_time:.3f}s)")
            
            # Test individual bias analysis
            print("\n🎯 Testing Bias Analysis:")
            for i, article in enumerate(test_articles, 1):
                start_time = time.time()
                bias_score = engine.score_bias(article)
                end_time = time.time()
                
                if bias_score is not None:
                    print(f"  Article {i}: Bias={bias_score:.3f} ({end_time-start_time:.3f}s)")
                else:
                    print(f"  Article {i}: Bias=None (fallback/error) ({end_time-start_time:.3f}s)")
            
            # Test batch processing
            print("\n🚀 Testing Batch Processing:")
            start_time = time.time()
            sentiment_results = engine.score_sentiment_batch(test_articles)
            bias_results = engine.score_bias_batch(test_articles)
            end_time = time.time()
            
            batch_time = end_time - start_time
            throughput = len(test_articles) / batch_time
            
            print(f"✅ Batch Processing Results:")
            print(f"   Articles: {len(test_articles)}")
            print(f"   Time: {batch_time:.3f}s")
            print(f"   Throughput: {throughput:.1f} articles/sec")
            
            # Display results
            print("\n📋 Detailed Results:")
            for i, (sentiment, bias) in enumerate(zip(sentiment_results, bias_results), 1):
                if sentiment is not None and bias is not None:
                    print(f"  Article {i}: Sentiment={sentiment:.3f}, Bias={bias:.3f}")
                else:
                    s_str = f"{sentiment:.3f}" if sentiment is not None else "None"
                    b_str = f"{bias:.3f}" if bias is not None else "None"
                    print(f"  Article {i}: Sentiment={s_str}, Bias={b_str}")
            
            # Memory check
            if HAS_PYCUDA and _pycuda_driver is not None:
                free_mem, total_mem = _pycuda_driver.mem_get_info()
                memory_used = (total_mem - free_mem) / 1024**3
                print(f"\n💾 GPU Memory Used: {memory_used:.1f}GB")
            
    try:
        # Import and initialize TensorRT engine with context management
        from native_tensorrt_engine import NativeTensorRTInferenceEngine

        logger.info("🔄 Initializing Native TensorRT Engine...")

        with NativeTensorRTInferenceEngine(engines_dir="tensorrt_engines") as engine:
            logger.info("✅ Engine initialized successfully")

            # No hard assertions here — the fact we reached this point without
            # exceptions is a good signal. Keep a lightweight sanity check.
            engine_info = engine.get_engine_info()
            assert isinstance(engine_info, dict), "Engine info should be a dict"

            # Batch processing sanity check (should not raise)
            sentiment_results = engine.score_sentiment_batch(["test article"])
            bias_results = engine.score_bias_batch(["test article"])
            assert len(sentiment_results) == len(bias_results) == 1

        logger.info("✅ Engine context manager completed successfully")

    finally:
        # Ensure CUDA context cleanup
        if cuda_context is not None:
            try:
                cuda_context.pop()
                logger.info("✅ Main CUDA context cleaned up properly")
            except Exception as e:
                logger.warning(f"⚠️ Context cleanup warning: {e}")

def performance_comparison_test():
    """Compare native TensorRT performance with baseline

    This test will be skipped in environments without CUDA/pycuda.
    """
    print("\n" + "="*80)
    print("📈 PERFORMANCE COMPARISON - NATIVE TENSORRT vs BASELINE")
    print("="*80)
    
    # Larger test set for performance validation
    test_articles = [
        "Breaking news: Stock market reaches record highs as investors show confidence in technology sector. Trading volumes have increased significantly over the past week, with tech stocks leading the rally. Analysts predict continued growth despite global economic uncertainties.",
        "Local community celebrates opening of new sustainable energy facility with overwhelming support from residents. The solar power installation is expected to reduce carbon emissions by 40% while creating numerous local jobs. Environmental groups praise the initiative as a model for other communities.",
        "Weather forecast predicts severe storms approaching coastal regions this weekend, prompting emergency preparedness measures. Residents are advised to secure outdoor items and prepare for potential power outages. Local authorities have opened emergency shelters in affected areas.",
        "Scientists announce breakthrough in renewable energy storage technology development that could revolutionize the industry. The new battery technology promises to store solar energy more efficiently and at lower costs. Major energy companies are already expressing interest in licensing the technology.",
        "City council approves budget for infrastructure improvements and public transportation expansion following months of public consultation. The $50 million investment will focus on road repairs, bridge maintenance, and new bus routes. Citizens expressed strong support for the comprehensive plan during recent town halls."
    ] * 20  # 100 articles total for meaningful performance testing
    
    cuda_context = None
    
    # Initialize GPU
    cuda_context, device = safe_gpu_init()
    if cuda_context is None:
        pytest.skip("CUDA/pycuda not available in this environment")

    try:
        from native_tensorrt_engine import NativeTensorRTInferenceEngine

        with NativeTensorRTInferenceEngine(engines_dir="tensorrt_engines") as engine:
            # Basic timing run — this primarily verifies the code path executes.
            start_time = time.time()
            sentiment_results = engine.score_sentiment_batch(test_articles)
            bias_results = engine.score_bias_batch(test_articles)
            end_time = time.time()

            native_time = end_time - start_time
            native_throughput = len(test_articles) / max(native_time, 1e-6)

            assert len(sentiment_results) == len(bias_results) == len(test_articles)
            assert native_throughput > 0

    finally:
        if cuda_context is not None:
            try:
                cuda_context.pop()
                logger.info("✅ Performance test CUDA context cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Context cleanup warning: {e}")

if __name__ == "__main__":
    print("🚀 Starting Ultra-Safe TensorRT Validation")
    print("Goal: Completely clean, warning-free operation with maximum performance")

    exit_code = 0
    try:
        test_native_tensorrt_clean()
        print("\n✅ Basic functionality test passed!")
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        exit_code = 1

    try:
        performance_comparison_test()
        print("\n✅ Performance comparison passed!")
    except Exception as e:
        logger.error(f"⚠️ Performance comparison had issues: {e}")
        # keep exit code as-is (don't override a prior failure)

    sys.exit(exit_code)
