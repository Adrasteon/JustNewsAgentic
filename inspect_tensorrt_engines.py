#!/usr/bin/env python3
"""
TensorRT Engine Inspection Tool
==============================

Inspect the actual tensor names and bindings in our compiled engines
to identify the root cause of the tensor binding errors.
"""

def inspect_tensorrt_engines():
    """Inspect the compiled TensorRT engines"""
    print("🔍 TENSORRT ENGINE INSPECTION")
    print("=" * 50)
    
    try:
        import tensorrt as trt
        from pathlib import Path
        
        engines_dir = Path("agents/analyst/tensorrt_engines")
        
        # Inspect sentiment engine
        sentiment_engine_path = engines_dir / "native_sentiment_roberta.engine"
        if sentiment_engine_path.exists():
            print("\n📊 SENTIMENT ENGINE ANALYSIS:")
            with open(sentiment_engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(engine_data)
            
            print(f"   Engine Name: Unnamed Network 0")
            print(f"   Num IO Tensors: {engine.num_io_tensors}")
            
            print("   All Tensors:")
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                mode = engine.get_tensor_mode(name)
                shape = engine.get_tensor_shape(name)
                dtype = engine.get_tensor_dtype(name)
                print(f"     [{i}] '{name}': {mode} - Shape: {shape}, Type: {dtype}")
                
            # Check for specific tensors we're trying to bind
            expected_tensors = ['input_ids', 'attention_mask', 'logits']
            print(f"   Expected Tensor Check:")
            for tensor_name in expected_tensors:
                try:
                    mode = engine.get_tensor_mode(tensor_name)
                    shape = engine.get_tensor_shape(tensor_name)
                    print(f"     ✅ '{tensor_name}': {mode} - {shape}")
                except:
                    print(f"     ❌ '{tensor_name}': NOT FOUND")
        
        # Inspect bias engine
        bias_engine_path = engines_dir / "native_bias_bert.engine"
        if bias_engine_path.exists():
            print("\n📊 BIAS ENGINE ANALYSIS:")
            with open(bias_engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(engine_data)
            
            print(f"   Engine Name: Unnamed Network 0")
            print(f"   Num IO Tensors: {engine.num_io_tensors}")
            
            print("   All Tensors:")
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                mode = engine.get_tensor_mode(name)
                shape = engine.get_tensor_shape(name)
                dtype = engine.get_tensor_dtype(name)
                print(f"     [{i}] '{name}': {mode} - Shape: {shape}, Type: {dtype}")
                
            # Check for specific tensors we're trying to bind
            expected_tensors = ['input_ids', 'attention_mask', 'logits']
            print(f"   Expected Tensor Check:")
            for tensor_name in expected_tensors:
                try:
                    mode = engine.get_tensor_mode(tensor_name)
                    shape = engine.get_tensor_shape(tensor_name)
                    print(f"     ✅ '{tensor_name}': {mode} - {shape}")
                except:
                    print(f"     ❌ '{tensor_name}': NOT FOUND")
        
        print(f"\n🎯 DIAGNOSIS:")
        print("Look for:")
        print("- Missing tensor bindings (we might need more than input_ids, attention_mask, logits)")
        print("- Incorrect tensor names (input.3 suggests different naming)")
        print("- Shape mismatches between expected and actual")
        
    except Exception as e:
        print(f"❌ Inspection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_tensorrt_engines()
