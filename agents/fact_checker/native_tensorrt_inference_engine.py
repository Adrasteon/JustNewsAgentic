#!/usr/bin/env python3
"""
Native TensorRT Inference Engine for Fact Checker Agent
Ultra-high performance inference using compiled TensorRT engines

Handles 2 TensorRT-optimized models + 2 fallback models:
- DistilBERT fact verification → TensorRT
- RoBERTa credibility assessment → TensorRT  
- SentenceTransformers evidence retrieval → GPU fallback
- spaCy NER claim extraction → CPU/GPU fallback

Target performance: 600+ articles/sec for fact checking
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

class NativeTensorRTFactCheckerEngine:
    """
    Ultra-high performance TensorRT inference engine for Fact Checker V2
    Uses compiled native TensorRT engines for maximum speed
    
    Handles hybrid architecture:
    - 2 TensorRT engines (fact verification, credibility assessment)
    - 2 fallback models (evidence retrieval, claim extraction)
    """
    
    def __init__(self, engines_dir: str = "agents/fact_checker/tensorrt_engines", fallback_fact_checker=None):
        """Initialize native TensorRT inference engine with fallback support"""
        self.engines_dir = Path(engines_dir)
        self.engines = {}
        self.contexts = {}
        self.tokenizers = {}
        self.engine_metadata = {}
        self.cuda_stream = None
        self.cuda_context = None
        self.fallback_fact_checker = fallback_fact_checker
        
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
                    logger.info("✅ Created new CUDA context for Fact Checker TensorRT")
                else:
                    logger.info("✅ Using existing CUDA context for Fact Checker TensorRT")
            except cuda.LogicError:
                # No context exists, create one
                device = cuda.Device(0)
                self.cuda_context = device.make_context()
                context_created = True
                logger.info("✅ Created new CUDA context for Fact Checker TensorRT")
            
            # Store whether we created the context (for cleanup)
            self.context_created = context_created
            
            # Create stream with proper context
            self.cuda_stream = cuda.Stream()
            
            # Verify stream is valid
            if self.cuda_stream.handle == 0:
                raise RuntimeError("Invalid CUDA stream handle")
            
            logger.info("✅ CUDA context initialized for Fact Checker native TensorRT")
            logger.info(f"   Stream handle: {self.cuda_stream.handle}")
            
        except Exception as e:
            logger.error(f"❌ CUDA initialization failed: {e}")
            logger.info("   Will use fallback Fact Checker engine")
            
    def _load_engines(self):
        """Load all available TensorRT engines for Fact Checker models"""
        if not self.engines_dir.exists():
            logger.warning(f"⚠️ TensorRT engines directory not found: {self.engines_dir}")
            logger.info("   Will use fallback Fact Checker engine")
            return
        
        # Expected Fact Checker TensorRT engines
        expected_engines = {
            "fact_verification": "native_fact_verification.engine",
            "credibility_assessment": "native_credibility_assessment.engine"
            # Note: evidence_retrieval and claim_extraction handled by fallback
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
                logger.info(f"🚀 Fact Checker Native TensorRT ready with {len(self.engines)} engines")
                logger.info(f"   Available engines: {list(self.engines.keys())}")
                logger.info("   Target performance: 600+ articles/sec")
            else:
                logger.warning("⚠️ No TensorRT engines loaded - will use fallback")
                
        except ImportError:
            logger.warning("⚠️ TensorRT not available - will use fallback Fact Checker engine")
        except Exception as e:
            logger.error(f"❌ Engine loading failed: {e}")

    def verify_fact(self, claim: str) -> Dict[str, Any]:
        """
        Verify a factual claim using native TensorRT or fallback
        
        Args:
            claim: Factual claim to verify
            
        Returns:
            {"verification": str, "confidence": float, "method": str}
        """
        start_time = time.time()
        
        if "fact_verification" in self.engines:
            try:
                # Use native TensorRT
                result = self._run_tensorrt_classification(
                    claim, "fact_verification", ["questionable", "factual"]
                )
                
                self.performance_stats['native_requests'] += 1
                self.performance_stats['native_time'] += time.time() - start_time
                
                return {
                    "verification": result['predicted_class'],
                    "confidence": result['confidence'],
                    "method": "native_tensorrt",
                    "inference_time_ms": (time.time() - start_time) * 1000
                }
                
            except Exception as e:
                logger.error(f"❌ TensorRT fact verification failed: {e}")
                # Fall through to fallback
        
        # Use fallback Fact Checker engine
        if self.fallback_fact_checker:
            try:
                self.performance_stats['fallback_requests'] += 1
                self.performance_stats['fallback_time'] += time.time() - start_time
                
                result = self.fallback_fact_checker.verify_fact(claim)
                result["method"] = "fallback_gpu"
                result["inference_time_ms"] = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                logger.error(f"❌ Fallback fact verification failed: {e}")
        
        # Default response
        return {
            "verification": "questionable",
            "confidence": 0.0,
            "method": "error",
            "inference_time_ms": (time.time() - start_time) * 1000
        }

    def assess_credibility(self, source_text: str) -> Dict[str, Any]:
        """
        Assess source credibility using native TensorRT or fallback
        
        Args:
            source_text: Source text to assess
            
        Returns:
            {"credibility": str, "confidence": float, "method": str}
        """
        start_time = time.time()
        
        if "credibility_assessment" in self.engines:
            try:
                # Use native TensorRT
                result = self._run_tensorrt_classification(
                    source_text, "credibility_assessment", ["low", "medium", "high"]
                )
                
                self.performance_stats['native_requests'] += 1
                self.performance_stats['native_time'] += time.time() - start_time
                
                return {
                    "credibility": result['predicted_class'],
                    "confidence": result['confidence'],
                    "method": "native_tensorrt",
                    "inference_time_ms": (time.time() - start_time) * 1000
                }
                
            except Exception as e:
                logger.error(f"❌ TensorRT credibility assessment failed: {e}")
                # Fall through to fallback
        
        # Use fallback Fact Checker engine
        if self.fallback_fact_checker:
            try:
                self.performance_stats['fallback_requests'] += 1
                self.performance_stats['fallback_time'] += time.time() - start_time
                
                result = self.fallback_fact_checker.assess_credibility(source_text)
                result["method"] = "fallback_gpu"
                result["inference_time_ms"] = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                logger.error(f"❌ Fallback credibility assessment failed: {e}")
        
        # Default response
        return {
            "credibility": "medium",
            "confidence": 0.0,
            "method": "error",
            "inference_time_ms": (time.time() - start_time) * 1000
        }

    def retrieve_evidence(self, query: str, sources: List[str] = None) -> Dict[str, Any]:
        """
        Retrieve evidence using SentenceTransformers (always fallback)
        
        Args:
            query: Query for evidence retrieval
            sources: Optional list of source texts
            
        Returns:
            {"evidence": list, "scores": list, "method": str}
        """
        start_time = time.time()
        
        # Always use fallback for evidence retrieval (SentenceTransformers)
        if self.fallback_fact_checker:
            try:
                self.performance_stats['fallback_requests'] += 1
                self.performance_stats['fallback_time'] += time.time() - start_time
                
                result = self.fallback_fact_checker.retrieve_evidence(query, sources)
                result["method"] = "fallback_sentence_transformers"
                result["inference_time_ms"] = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                logger.error(f"❌ Evidence retrieval failed: {e}")
        
        # Default response
        return {
            "evidence": [],
            "scores": [],
            "method": "error",
            "inference_time_ms": (time.time() - start_time) * 1000
        }

    def extract_claims(self, text: str) -> Dict[str, Any]:
        """
        Extract verifiable claims using spaCy NER (always fallback)
        
        Args:
            text: Text to extract claims from
            
        Returns:
            {"claims": list, "entities": list, "method": str}
        """
        start_time = time.time()
        
        # Always use fallback for claim extraction (spaCy NER)
        if self.fallback_fact_checker:
            try:
                self.performance_stats['fallback_requests'] += 1
                self.performance_stats['fallback_time'] += time.time() - start_time
                
                result = self.fallback_fact_checker.extract_claims(text)
                result["method"] = "fallback_spacy_ner"
                result["inference_time_ms"] = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                logger.error(f"❌ Claim extraction failed: {e}")
        
        # Default response
        return {
            "claims": [],
            "entities": [],
            "method": "error",
            "inference_time_ms": (time.time() - start_time) * 1000
        }

    def comprehensive_fact_check(self, article_text: str) -> Dict[str, Any]:
        """
        Perform comprehensive fact checking combining all methods
        
        Args:
            article_text: Article text to fact check
            
        Returns:
            Combined results from all fact checking methods
        """
        start_time = time.time()
        
        results = {
            "article_length": len(article_text),
            "methods_used": [],
            "processing_time_ms": 0
        }
        
        # Extract claims
        claims_result = self.extract_claims(article_text)
        results["claims"] = claims_result["claims"]
        results["methods_used"].append(claims_result["method"])
        
        # Verify main claims
        if claims_result["claims"]:
            main_claim = claims_result["claims"][0] if claims_result["claims"] else article_text[:500]
            verification_result = self.verify_fact(main_claim)
            results["verification"] = verification_result["verification"]
            results["verification_confidence"] = verification_result["confidence"]
            results["methods_used"].append(verification_result["method"])
        
        # Assess overall credibility
        credibility_result = self.assess_credibility(article_text)
        results["credibility"] = credibility_result["credibility"]
        results["credibility_confidence"] = credibility_result["confidence"]
        results["methods_used"].append(credibility_result["method"])
        
        # Retrieve supporting evidence
        if claims_result["claims"]:
            evidence_result = self.retrieve_evidence(claims_result["claims"][0] if claims_result["claims"] else article_text[:200])
            results["evidence_count"] = len(evidence_result.get("evidence", []))
            results["methods_used"].append(evidence_result["method"])
        
        results["processing_time_ms"] = (time.time() - start_time) * 1000
        results["methods_used"] = list(set(results["methods_used"]))  # Remove duplicates
        
        return results

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
            "hybrid_architecture": {
                "tensorrt_models": list(self.engines.keys()),
                "fallback_models": ["evidence_retrieval", "claim_extraction"]
            },
            "performance_improvement": (fallback_avg_time / native_avg_time if native_avg_time > 0 else 0)
        }

    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up Fact Checker TensorRT engine...")
        
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
    """Test Fact Checker TensorRT engine"""
    print("🚀 Testing Fact Checker Native TensorRT Engine")
    print("==============================================")
    
    # Test with fallback support
    try:
        from fact_checker_v2_engine import FactCheckerV2Engine
        fallback_fact_checker = FactCheckerV2Engine(enable_training=False)
        print("✅ Fallback Fact Checker engine loaded")
    except Exception as e:
        print(f"⚠️ Could not load fallback Fact Checker engine: {e}")
        fallback_fact_checker = None
    
    with NativeTensorRTFactCheckerEngine(fallback_fact_checker=fallback_fact_checker) as fact_checker:
        # Test article
        test_article = """
        According to a new study published in Nature, researchers have discovered that 
        drinking green tea can reduce the risk of heart disease by up to 30%. The study 
        followed 10,000 participants over 5 years and found significant improvements 
        in cardiovascular health among regular green tea consumers.
        """
        
        print(f"\nTest article: {test_article[:100]}...")
        
        # Test all Fact Checker functions
        print("\n🔍 Testing Fact Verification:")
        verification_result = fact_checker.verify_fact("Green tea reduces heart disease risk by 30%")
        print(f"   Result: {verification_result}")
        
        print("\n📊 Testing Credibility Assessment:")
        credibility_result = fact_checker.assess_credibility(test_article)
        print(f"   Result: {credibility_result}")
        
        print("\n🔎 Testing Claim Extraction:")
        claims_result = fact_checker.extract_claims(test_article)
        print(f"   Result: {claims_result}")
        
        print("\n🧠 Testing Evidence Retrieval:")
        evidence_result = fact_checker.retrieve_evidence("green tea cardiovascular benefits")
        print(f"   Result: {evidence_result}")
        
        print("\n🎯 Testing Comprehensive Fact Check:")
        comprehensive_result = fact_checker.comprehensive_fact_check(test_article)
        print(f"   Result: {comprehensive_result}")
        
        # Performance stats
        print("\n📈 Performance Statistics:")
        stats = fact_checker.get_performance_stats()
        print(f"   {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    main()