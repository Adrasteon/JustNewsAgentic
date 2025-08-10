#!/usr/bin/env python3
"""
Native TensorRT Inference Engine for Scout Agent
Ultra-high performance inference using compiled TensorRT engines

Based on successful Analyst agent implementation
Target performance: 800+ articles/sec for 5-model Scout architecture
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json
import os
import sys

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NativeTensorRTScoutEngine:
    """
    Ultra-high performance TensorRT inference engine for Scout V2
    Uses compiled native TensorRT engines for maximum speed
    
    Handles 5-model architecture:
    - BERT news classifier
    - BERT quality assessor  
    - RoBERTa sentiment analyzer
    - RoBERTa bias detector
    - LLaVA visual analyzer (fallback to GPU)
    """
    
    def __init__(self, engines_dir: str = "agents/scout/tensorrt_engines", fallback_scout=None):
        """Initialize native TensorRT inference engine with fallback support"""
        self.engines_dir = Path(engines_dir)
        self.engines = {}
        self.contexts = {}
        self.tokenizers = {}
        self.engine_metadata = {}
        self.cuda_stream = None
        self.cuda_context = None
        self.fallback_scout = fallback_scout
        
        # Performance tracking
        self.performance_stats = {
            'native_requests': 0,
            'native_time': 0.0,
            'fallback_requests': 0,
            'fallback_time': 0.0
        }
        
        self.context_created = False
        self._initialize_cuda()
        self._load_engines()
    
    def _initialize_cuda(self):
        """Initialize CUDA context and stream with proper context management"""
        try:
            import pycuda.driver as cuda
            
            # Initialize CUDA explicitly (don't use autoinit)
            cuda.init()
            
            # Get or create CUDA context properly
            context_created = False
            try:
                # Try to get existing context
                self.cuda_context = cuda.Context.get_current()
                if self.cuda_context is None:
                    # Create new context if none exists
                    device = cuda.Device(0)
                    self.cuda_context = device.make_context()
                    context_created = True
                    logger.info("✅ Created new CUDA context for Scout TensorRT")
                else:
                    logger.info("✅ Using existing CUDA context for Scout TensorRT")
            except cuda.LogicError:
                # No context exists, create one
                device = cuda.Device(0)
                self.cuda_context = device.make_context()
                context_created = True
                logger.info("✅ Created new CUDA context for Scout TensorRT")
            
            # Store whether we created the context (for cleanup)
            self.context_created = context_created
            
            # Create stream with proper context
            self.cuda_stream = cuda.Stream()
            
            # Verify stream is valid
            if self.cuda_stream.handle == 0:
                raise RuntimeError("Invalid CUDA stream handle")
            
            logger.info("✅ CUDA context initialized for Scout native TensorRT")
            logger.info(f"   Stream handle: {self.cuda_stream.handle}")
            
        except Exception as e:
            logger.error(f"❌ CUDA initialization failed: {e}")
            logger.info("   Will use fallback Scout engine")
            
    def _load_engines(self):
        """Load all available TensorRT engines for Scout models"""
        if not self.engines_dir.exists():
            logger.warning(f"⚠️ TensorRT engines directory not found: {self.engines_dir}")
            logger.info("   Will use fallback Scout engine")
            return
        
        # Expected Scout TensorRT engines
        expected_engines = {
            "news_classifier": "native_news_classifier.engine",
            "quality_assessor": "native_quality_assessor.engine", 
            "sentiment_analyzer": "native_sentiment_analyzer.engine",
            "bias_detector": "native_bias_detector.engine"
            # LLaVA visual analyzer handled separately (may not be suitable for TensorRT)
        }
        
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            
            for engine_name, engine_file in expected_engines.items():
                engine_path = self.engines_dir / engine_file
                metadata_path = self.engines_dir / f"native_{engine_name}.json"
                
                if engine_path.exists() and metadata_path.exists():
                    try:
                        # Load engine
                        logger.info(f"🔧 Loading {engine_name} TensorRT engine...")
                        with open(engine_path, 'rb') as f:
                            engine_data = f.read()
                        
                        engine = runtime.deserialize_cuda_engine(engine_data)
                        if engine is None:
                            logger.error(f"❌ Failed to deserialize {engine_name} engine")
                            continue
                        
                        # Create execution context
                        context = engine.create_execution_context()
                        if context is None:
                            logger.error(f"❌ Failed to create execution context for {engine_name}")
                            continue
                        
                        # Load metadata
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Load tokenizer
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(metadata['model_name'])
                        
                        # Store components
                        self.engines[engine_name] = engine
                        self.contexts[engine_name] = context
                        self.tokenizers[engine_name] = tokenizer
                        self.engine_metadata[engine_name] = metadata
                        
                        engine_size_mb = len(engine_data) / (1024 * 1024)
                        logger.info(f"✅ {engine_name} engine loaded ({engine_size_mb:.1f} MB)")
                        logger.info(f"   Model: {metadata.get('model_name', 'unknown')}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to load {engine_name}: {e}")
                        continue
                else:
                    logger.warning(f"⚠️ {engine_name} engine not found: {engine_path}")
            
            if self.engines:
                logger.info(f"🚀 Scout Native TensorRT ready with {len(self.engines)} engines")
                logger.info(f"   Available engines: {list(self.engines.keys())}")
                logger.info("   Target performance: 800+ articles/sec")
            else:
                logger.warning("⚠️ No TensorRT engines loaded - will use fallback")
                
        except ImportError:
            logger.warning("⚠️ TensorRT not available - will use fallback Scout engine")
        except Exception as e:
            logger.error(f"❌ Engine loading failed: {e}")

    def classify_news(self, text: str) -> Dict[str, Any]:
        """
        Classify if text is news using native TensorRT or fallback
        
        Returns:
            {"is_news": bool, "confidence": float, "method": str}
        """
        start_time = time.time()
        
        if "news_classifier" in self.engines:
            try:
                # Use native TensorRT
                result = self._run_tensorrt_classification(
                    text, "news_classifier", ["not_news", "news"]
                )
                
                self.performance_stats['native_requests'] += 1
                self.performance_stats['native_time'] += time.time() - start_time
                
                return {
                    "is_news": result['predicted_class'] == 'news',
                    "confidence": result['confidence'],
                    "method": "native_tensorrt",
                    "inference_time_ms": (time.time() - start_time) * 1000
                }
                
            except Exception as e:
                logger.error(f"❌ TensorRT news classification failed: {e}")
                # Fall through to fallback
        
        # Use fallback Scout engine
        if self.fallback_scout:
            try:
                self.performance_stats['fallback_requests'] += 1
                self.performance_stats['fallback_time'] += time.time() - start_time
                
                result = self.fallback_scout.classify_news(text)
                result["method"] = "fallback_gpu"
                result["inference_time_ms"] = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                logger.error(f"❌ Fallback news classification failed: {e}")
        
        # Default response
        return {
            "is_news": False,
            "confidence": 0.0,
            "method": "error",
            "inference_time_ms": (time.time() - start_time) * 1000
        }

    def assess_quality(self, text: str) -> Dict[str, Any]:
        """
        Assess content quality using native TensorRT or fallback
        
        Returns:
            {"quality": str, "confidence": float, "method": str}
        """
        start_time = time.time()
        
        if "quality_assessor" in self.engines:
            try:
                # Use native TensorRT
                result = self._run_tensorrt_classification(
                    text, "quality_assessor", ["low", "medium", "high"]
                )
                
                self.performance_stats['native_requests'] += 1
                self.performance_stats['native_time'] += time.time() - start_time
                
                return {
                    "quality": result['predicted_class'],
                    "confidence": result['confidence'],
                    "method": "native_tensorrt",
                    "inference_time_ms": (time.time() - start_time) * 1000
                }
                
            except Exception as e:
                logger.error(f"❌ TensorRT quality assessment failed: {e}")
                # Fall through to fallback
        
        # Use fallback Scout engine
        if self.fallback_scout:
            try:
                self.performance_stats['fallback_requests'] += 1
                self.performance_stats['fallback_time'] += time.time() - start_time
                
                result = self.fallback_scout.assess_quality(text)
                result["method"] = "fallback_gpu"
                result["inference_time_ms"] = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                logger.error(f"❌ Fallback quality assessment failed: {e}")
        
        # Default response
        return {
            "quality": "medium",
            "confidence": 0.0,
            "method": "error",
            "inference_time_ms": (time.time() - start_time) * 1000
        }

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using native TensorRT or fallback
        
        Returns:
            {"sentiment": str, "confidence": float, "method": str}
        """
        start_time = time.time()
        
        if "sentiment_analyzer" in self.engines:
            try:
                # Use native TensorRT
                result = self._run_tensorrt_classification(
                    text, "sentiment_analyzer", ["negative", "neutral", "positive"]
                )
                
                self.performance_stats['native_requests'] += 1
                self.performance_stats['native_time'] += time.time() - start_time
                
                return {
                    "sentiment": result['predicted_class'],
                    "confidence": result['confidence'],
                    "method": "native_tensorrt",
                    "inference_time_ms": (time.time() - start_time) * 1000
                }
                
            except Exception as e:
                logger.error(f"❌ TensorRT sentiment analysis failed: {e}")
                # Fall through to fallback
        
        # Use fallback Scout engine
        if self.fallback_scout:
            try:
                self.performance_stats['fallback_requests'] += 1
                self.performance_stats['fallback_time'] += time.time() - start_time
                
                result = self.fallback_scout.analyze_sentiment(text)
                result["method"] = "fallback_gpu"
                result["inference_time_ms"] = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                logger.error(f"❌ Fallback sentiment analysis failed: {e}")
        
        # Default response
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "method": "error",
            "inference_time_ms": (time.time() - start_time) * 1000
        }

    def detect_bias(self, text: str) -> Dict[str, Any]:
        """
        Detect bias using native TensorRT or fallback
        
        Returns:
            {"bias": str, "confidence": float, "method": str}
        """
        start_time = time.time()
        
        if "bias_detector" in self.engines:
            try:
                # Use native TensorRT
                result = self._run_tensorrt_classification(
                    text, "bias_detector", ["not_biased", "biased"]
                )
                
                self.performance_stats['native_requests'] += 1
                self.performance_stats['native_time'] += time.time() - start_time
                
                return {
                    "bias": result['predicted_class'],
                    "confidence": result['confidence'],
                    "method": "native_tensorrt",
                    "inference_time_ms": (time.time() - start_time) * 1000
                }
                
            except Exception as e:
                logger.error(f"❌ TensorRT bias detection failed: {e}")
                # Fall through to fallback
        
        # Use fallback Scout engine
        if self.fallback_scout:
            try:
                self.performance_stats['fallback_requests'] += 1
                self.performance_stats['fallback_time'] += time.time() - start_time
                
                result = self.fallback_scout.detect_bias(text)
                result["method"] = "fallback_gpu"
                result["inference_time_ms"] = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                logger.error(f"❌ Fallback bias detection failed: {e}")
        
        # Default response
        return {
            "bias": "not_biased",
            "confidence": 0.0,
            "method": "error",
            "inference_time_ms": (time.time() - start_time) * 1000
        }

    def _run_tensorrt_classification(self, text: str, engine_name: str, class_labels: List[str]) -> Dict[str, Any]:
        """
        Run TensorRT inference for classification tasks
        
        Args:
            text: Input text
            engine_name: Name of the TensorRT engine
            class_labels: List of class labels in order
            
        Returns:
            {"predicted_class": str, "confidence": float, "logits": list}
        """
        import pycuda.driver as cuda
        import numpy as np
        
        engine = self.engines[engine_name]
        context = self.contexts[engine_name]
        tokenizer = self.tokenizers[engine_name]
        metadata = self.engine_metadata[engine_name]
        
        # Tokenize input
        max_length = metadata.get('max_sequence_length', 512)
        inputs = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        # Prepare input arrays
        input_ids = inputs['input_ids'].astype(np.int32)
        attention_mask = inputs['attention_mask'].astype(np.int32)
        
        # Get expected output shape
        num_labels = metadata.get('num_labels', len(class_labels))
        output_shape = (1, num_labels)
        
        # Allocate GPU memory
        input_ids_gpu = cuda.mem_alloc(input_ids.nbytes)
        attention_mask_gpu = cuda.mem_alloc(attention_mask.nbytes)
        output_gpu = cuda.mem_alloc(np.empty(output_shape, dtype=np.float32).nbytes)
        
        # Copy input data to GPU
        cuda.memcpy_htod_async(input_ids_gpu, input_ids, self.cuda_stream)
        cuda.memcpy_htod_async(attention_mask_gpu, attention_mask, self.cuda_stream)
        
        # Set input shapes (for dynamic batching)
        context.set_input_shape("input_ids", input_ids.shape)
        context.set_input_shape("attention_mask", attention_mask.shape)
        
        # Set tensor addresses
        context.set_tensor_address("input_ids", int(input_ids_gpu))
        context.set_tensor_address("attention_mask", int(attention_mask_gpu))
        context.set_tensor_address("logits", int(output_gpu))
        
        # Run inference
        context.execute_async_v3(self.cuda_stream.handle)
        
        # Copy output back to host
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, output_gpu, self.cuda_stream)
        self.cuda_stream.synchronize()
        
        # Process results
        logits = output[0]  # Remove batch dimension
        probabilities = self._softmax(logits)
        predicted_idx = np.argmax(probabilities)
        predicted_class = class_labels[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Cleanup GPU memory
        input_ids_gpu.free()
        attention_mask_gpu.free()
        output_gpu.free()
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "logits": logits.tolist()
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits"""
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_requests = self.performance_stats['native_requests'] + self.performance_stats['fallback_requests']
        
        if total_requests == 0:
            return {"status": "no_requests"}
        
        native_avg_time = (self.performance_stats['native_time'] / self.performance_stats['native_requests'] 
                          if self.performance_stats['native_requests'] > 0 else 0)
        fallback_avg_time = (self.performance_stats['fallback_time'] / self.performance_stats['fallback_requests']
                            if self.performance_stats['fallback_requests'] > 0 else 0)
        
        return {
            "total_requests": total_requests,
            "native_requests": self.performance_stats['native_requests'],
            "fallback_requests": self.performance_stats['fallback_requests'],
            "native_percentage": (self.performance_stats['native_requests'] / total_requests) * 100,
            "native_avg_time_ms": native_avg_time * 1000,
            "fallback_avg_time_ms": fallback_avg_time * 1000,
            "engines_loaded": list(self.engines.keys()),
            "performance_improvement": (fallback_avg_time / native_avg_time if native_avg_time > 0 else 0)
        }

    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up Scout TensorRT engine...")
        
        # Clear contexts
        for context in self.contexts.values():
            try:
                del context
            except:
                pass
        self.contexts.clear()
        
        # Clear engines
        for engine in self.engines.values():
            try:
                del engine
            except:
                pass
        self.engines.clear()
        
        # Clean up CUDA context if we created it
        if self.context_created and self.cuda_context:
            try:
                import pycuda.driver as cuda
                self.cuda_context.pop()
                logger.info("✅ CUDA context cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ CUDA context cleanup warning: {e}")
        
        # Clear GPU memory
        try:
            import torch
            torch.cuda.empty_cache()
            logger.info("✅ GPU memory cache cleared")
        except:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def main():
    """Test Scout TensorRT engine"""
    print("🚀 Testing Scout Native TensorRT Engine")
    print("======================================")
    
    # Test with fallback support
    try:
        from gpu_scout_engine_v2 import NextGenGPUScoutEngine
        fallback_scout = NextGenGPUScoutEngine(enable_training=False)
        print("✅ Fallback Scout engine loaded")
    except Exception as e:
        print(f"⚠️ Could not load fallback Scout engine: {e}")
        fallback_scout = None
    
    with NativeTensorRTScoutEngine(fallback_scout=fallback_scout) as scout_engine:
        # Test news classification
        test_article = """
        Breaking News: Scientists at MIT have developed a revolutionary new battery technology 
        that could charge electric vehicles in under 60 seconds. The breakthrough uses novel 
        graphene-based electrodes and could transform the automotive industry.
        """
        
        print(f"\nTest article: {test_article[:100]}...")
        
        # Test all Scout functions
        print("\n🔍 Testing News Classification:")
        news_result = scout_engine.classify_news(test_article)
        print(f"   Result: {news_result}")
        
        print("\n📊 Testing Quality Assessment:")
        quality_result = scout_engine.assess_quality(test_article)
        print(f"   Result: {quality_result}")
        
        print("\n😊 Testing Sentiment Analysis:")
        sentiment_result = scout_engine.analyze_sentiment(test_article)
        print(f"   Result: {sentiment_result}")
        
        print("\n⚖️ Testing Bias Detection:")
        bias_result = scout_engine.detect_bias(test_article)
        print(f"   Result: {bias_result}")
        
        # Performance stats
        print("\n📈 Performance Statistics:")
        stats = scout_engine.get_performance_stats()
        print(f"   {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    main()