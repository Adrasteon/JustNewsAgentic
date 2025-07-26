#!/usr/bin/env python3
"""
TensorRT-LLM Installation Test Script for JustNews V4
Comprehensive testing of TensorRT-LLM with RTX 3090
"""

import sys
import os
import traceback

def test_basic_imports():
    """Test basic Python imports"""
    print("🔍 Testing basic imports...")
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        
        import numpy as np
        print(f"   ✅ NumPy {np.__version__}")
        
        return True
    except Exception as e:
        print(f"   ❌ Basic imports failed: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("🔍 Testing CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.version.cuda}")
            print(f"   ✅ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   ✅ GPU {i}: {props.name}")
                print(f"      VRAM: {props.total_memory / 1024**3:.1f} GB")
                print(f"      Compute: {props.major}.{props.minor}")
            return True
        else:
            print("   ❌ CUDA not available")
            return False
    except Exception as e:
        print(f"   ❌ CUDA test failed: {e}")
        return False

def test_mpi():
    """Test MPI availability"""
    print("🔍 Testing MPI...")
    try:
        from mpi4py import MPI
        print(f"   ✅ MPI4Py imported successfully")
        print(f"   ✅ MPI Version: {MPI.Get_version()}")
        return True
    except Exception as e:
        print(f"   ❌ MPI test failed: {e}")
        print("   💡 This might be expected in some environments")
        traceback.print_exc()
        return False

def test_tensorrt():
    """Test TensorRT availability"""
    print("🔍 Testing TensorRT...")
    try:
        import tensorrt as trt
        print(f"   ✅ TensorRT imported successfully")
        print(f"   ✅ TensorRT Version: {trt.__version__}")
        return True
    except Exception as e:
        print(f"   ❌ TensorRT test failed: {e}")
        return False

def test_tensorrt_llm():
    """Test TensorRT-LLM import"""
    print("🔍 Testing TensorRT-LLM...")
    try:
        # Try to import without MPI first
        os.environ['OMPI_MCA_plm'] = 'isolated'
        os.environ['OMPI_MCA_btl_vader_single_copy_mechanism'] = 'none'
        os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = '1'
        
        import tensorrt_llm
        print(f"   ✅ TensorRT-LLM imported successfully!")
        print(f"   ✅ Version: {tensorrt_llm.__version__}")
        
        # Test basic functionality
        from tensorrt_llm import Builder
        from tensorrt_llm.network import net_guard
        print("   ✅ Core TensorRT-LLM components available")
        
        return True
    except Exception as e:
        print(f"   ❌ TensorRT-LLM test failed: {e}")
        traceback.print_exc()
        return False

def test_transformers():
    """Test Transformers library"""
    print("🔍 Testing Transformers...")
    try:
        import transformers
        print(f"   ✅ Transformers {transformers.__version__}")
        return True
    except Exception as e:
        print(f"   ❌ Transformers test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 TensorRT-LLM Installation Test for JustNews V4")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("CUDA Support", test_cuda),
        ("MPI Support", test_mpi),
        ("TensorRT", test_tensorrt),
        ("Transformers", test_transformers),
        ("TensorRT-LLM", test_tensorrt_llm),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if results.get("TensorRT-LLM", False):
        print("\n🎉 TensorRT-LLM is ready for JustNews V4!")
        print("   Next steps:")
        print("   1. Test model loading")
        print("   2. Test inference performance")
        print("   3. Integrate with JustNews V4")
    else:
        print("\n⚠️  TensorRT-LLM needs additional setup")
        print("   Check the error messages above for guidance")
    
    return results.get("TensorRT-LLM", False)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
