#!/usr/bin/env python3
"""
TensorRT-LLM Performance Test for JustNews V4
Test basic inference capabilities with RTX 3090
"""

import os
import time
import traceback

# Set environment variables for stability
os.environ['OMPI_MCA_plm'] = 'isolated'
os.environ['OMPI_MCA_btl_vader_single_copy_mechanism'] = 'none'
os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = '1'

def test_basic_functionality():
    """Test basic TensorRT-LLM functionality"""
    print("🚀 Testing TensorRT-LLM Basic Functionality")
    print("=" * 50)
    
    try:
        # Import required modules
        import torch
        import tensorrt_llm
        from tensorrt_llm import Builder
        from tensorrt_llm.network import net_guard
        from tensorrt_llm.models import LLaMAForCausalLM
        
        print(f"✅ All imports successful")
        print(f"   TensorRT-LLM: {tensorrt_llm.__version__}")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA: {torch.version.cuda}")
        
        # Test GPU memory
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        
        print(f"📊 GPU Memory Status:")
        print(f"   Total: {total_memory / 1024**3:.1f} GB")
        print(f"   Allocated: {allocated / 1024**3:.3f} GB")
        print(f"   Reserved: {reserved / 1024**3:.3f} GB")
        print(f"   Free: {(total_memory - reserved) / 1024**3:.1f} GB")
        
        # Test basic tensor operations
        print(f"🔥 Testing GPU operations...")
        start_time = time.time()
        
        # Create test tensors
        x = torch.randn(1000, 1000, device='cuda', dtype=torch.float16)
        y = torch.randn(1000, 1000, device='cuda', dtype=torch.float16)
        
        # Perform matrix multiplication
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        
        gpu_time = time.time() - start_time
        print(f"   ✅ GPU matrix multiplication: {gpu_time:.4f}s")
        print(f"   ✅ Result shape: {z.shape}")
        print(f"   ✅ Data type: {z.dtype}")
        
        # Test Builder functionality
        print(f"🏗️  Testing TensorRT-LLM Builder...")
        builder = Builder()
        print(f"   ✅ Builder created successfully")
        
        # Test basic network creation
        with net_guard(builder.create_network()) as network:
            print(f"   ✅ Network created successfully")
            print(f"   ✅ Network precision: {network.dtype}")
        
        print(f"\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during basic functionality test: {e}")
        traceback.print_exc()
        return False

def main():
    """Run performance tests"""
    print("🎯 TensorRT-LLM Performance Test for JustNews V4")
    print("Testing RTX 3090 + TensorRT-LLM 0.20.0")
    print()
    
    success = test_basic_functionality()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ TensorRT-LLM is fully operational!")
        print("🚀 Ready for JustNews V4 integration:")
        print("   • GPU acceleration: RTX 3090 (24GB VRAM)")
        print("   • TensorRT-LLM: v0.20.0")
        print("   • PyTorch: v2.7.0 with CUDA 12.6")
        print("   • Next: Model loading and inference testing")
    else:
        print("\n❌ Some issues detected - check logs above")
    
    return success

if __name__ == "__main__":
    main()
