#!/usr/bin/env python3
"""
Native TensorRT Compiler for Scout Agent
Compiles Scout V2 models to optimized TensorRT engines

Based on successful Analyst agent TensorRT implementation
Target performance: 800+ articles/sec for 5-model architecture
"""

import logging
import os
import sys
from pathlib import Path
import json
import torch
from typing import Dict, List, Optional, Tuple

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScoutTensorRTCompiler:
    """
    Native TensorRT compiler for Scout V2 5-model architecture
    Compiles: BERT news classifier, BERT quality assessor, RoBERTa sentiment, 
              RoBERTa bias detector, and LLaVA visual analyzer
    """
    
    def __init__(self, engines_dir: str = "agents/scout/tensorrt_engines"):
        """Initialize Scout TensorRT compiler"""
        self.engines_dir = Path(engines_dir)
        self.engines_dir.mkdir(parents=True, exist_ok=True)
        
        # Scout V2 model configurations
        self.model_configs = {
            "news_classifier": {
                "model_name": "google-bert/bert-base-uncased",
                "task": "sequence_classification",
                "num_labels": 2,  # Binary: news/not-news
                "max_sequence_length": 512,
                "batch_size": 32,
                "precision": "fp16"
            },
            "quality_assessor": {
                "model_name": "google-bert/bert-base-uncased", 
                "task": "sequence_classification",
                "num_labels": 3,  # Low/Medium/High quality
                "max_sequence_length": 512,
                "batch_size": 16,
                "precision": "fp16"
            },
            "sentiment_analyzer": {
                "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "task": "sequence_classification",
                "num_labels": 3,  # Negative/Neutral/Positive
                "max_sequence_length": 512,
                "batch_size": 24,
                "precision": "fp16"
            },
            "bias_detector": {
                "model_name": "martin-ha/toxic-comment-model",
                "task": "sequence_classification", 
                "num_labels": 2,  # Binary: biased/not-biased
                "max_sequence_length": 512,
                "batch_size": 16,
                "precision": "fp16"
            }
            # Note: LLaVA visual analyzer requires special handling and may not be suitable for TensorRT
        }
        
        logger.info("✅ Scout TensorRT Compiler initialized")
        logger.info(f"   Engines directory: {self.engines_dir}")
        logger.info(f"   Models to compile: {len(self.model_configs)}")

    def compile_model(self, model_key: str) -> bool:
        """
        Compile a single Scout model to TensorRT engine
        
        Args:
            model_key: Key from model_configs
            
        Returns:
            True if compilation successful
        """
        try:
            config = self.model_configs[model_key]
            logger.info(f"🔄 Compiling {model_key} to TensorRT...")
            logger.info(f"   Model: {config['model_name']}")
            logger.info(f"   Precision: {config['precision']}")
            
            # Import TensorRT components
            try:
                import tensorrt as trt
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
            except ImportError as e:
                logger.error(f"❌ Missing dependencies: {e}")
                return False
            
            # Initialize TensorRT logger and builder
            trt_logger = trt.Logger(trt.Logger.INFO)
            builder = trt.Builder(trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # Load the model and tokenizer
            logger.info(f"   Loading {config['model_name']}...")
            tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
            model = AutoModelForSequenceClassification.from_pretrained(
                config['model_name'],
                num_labels=config.get('num_labels', 2)
            )
            model.eval()
            model.to('cuda')
            
            # Create ONNX export for TensorRT conversion
            dummy_input = {
                'input_ids': torch.randint(0, tokenizer.vocab_size, (1, config['max_sequence_length'])).cuda(),
                'attention_mask': torch.ones(1, config['max_sequence_length']).cuda()
            }
            
            onnx_path = self.engines_dir / f"{model_key}_temp.onnx"
            
            # Export to ONNX
            logger.info(f"   Exporting to ONNX: {onnx_path}")
            torch.onnx.export(
                model,
                tuple(dummy_input.values()),
                str(onnx_path),
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size'},
                    'attention_mask': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                },
                opset_version=11
            )
            
            # Parse ONNX to TensorRT network
            parser = trt.OnnxParser(network, trt_logger)
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    logger.error(f"❌ Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error(f"   {parser.get_error(error)}")
                    return False
            
            # Configure builder
            builder_config = builder.create_builder_config()
            builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace
            
            if config['precision'] == 'fp16':
                if builder.platform_has_fast_fp16:
                    builder_config.set_flag(trt.BuilderFlag.FP16)
                    logger.info("   ✅ FP16 precision enabled")
                else:
                    logger.warning("   ⚠️ FP16 not supported, using FP32")
            
            # Set optimization profile for dynamic batching
            profile = builder.create_optimization_profile()
            profile.set_shape(
                'input_ids',
                (1, config['max_sequence_length']),          # min
                (config['batch_size'] // 2, config['max_sequence_length']),  # opt  
                (config['batch_size'], config['max_sequence_length'])         # max
            )
            profile.set_shape(
                'attention_mask', 
                (1, config['max_sequence_length']),          # min
                (config['batch_size'] // 2, config['max_sequence_length']),  # opt
                (config['batch_size'], config['max_sequence_length'])         # max
            )
            builder_config.add_optimization_profile(profile)
            
            # Build TensorRT engine
            logger.info("   🔨 Building TensorRT engine...")
            serialized_engine = builder.build_serialized_network(network, builder_config)
            
            if serialized_engine is None:
                logger.error(f"❌ Failed to build TensorRT engine for {model_key}")
                return False
            
            # Save engine
            engine_path = self.engines_dir / f"native_{model_key}.engine"
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            
            # Save metadata
            metadata = {
                "model_name": config['model_name'],
                "task": config['task'],
                "num_labels": config.get('num_labels', 2),
                "max_sequence_length": config['max_sequence_length'],
                "batch_size": config['batch_size'],
                "precision": config['precision'],
                "input_shapes": {
                    "input_ids": [config['batch_size'], config['max_sequence_length']],
                    "attention_mask": [config['batch_size'], config['max_sequence_length']]
                },
                "output_shapes": {
                    "logits": [config['batch_size'], config.get('num_labels', 2)]
                },
                "compilation_date": str(torch.datetime.datetime.now()),
                "tensorrt_version": trt.__version__
            }
            
            metadata_path = self.engines_dir / f"native_{model_key}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Cleanup temporary ONNX file
            os.unlink(onnx_path)
            
            # Report success
            engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
            logger.info(f"✅ {model_key} compiled successfully")
            logger.info(f"   Engine: {engine_path} ({engine_size_mb:.1f} MB)")
            logger.info(f"   Metadata: {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Compilation failed for {model_key}: {e}")
            return False

    def compile_all_models(self) -> Dict[str, bool]:
        """
        Compile all Scout V2 models to TensorRT engines
        
        Returns:
            Dictionary of compilation results
        """
        logger.info("🚀 Starting Scout V2 TensorRT compilation...")
        logger.info(f"   Target performance: 800+ articles/sec")
        logger.info(f"   Models to compile: {list(self.model_configs.keys())}")
        
        results = {}
        
        for model_key in self.model_configs.keys():
            logger.info(f"\n=== Compiling {model_key.upper()} ===")
            results[model_key] = self.compile_model(model_key)
        
        # Report summary
        successful = sum(results.values())
        total = len(results)
        
        logger.info(f"\n🎯 SCOUT TENSORRT COMPILATION SUMMARY")
        logger.info(f"   Successful: {successful}/{total}")
        logger.info(f"   Failed: {total - successful}/{total}")
        
        if successful == total:
            logger.info("🎉 ALL SCOUT MODELS COMPILED SUCCESSFULLY!")
            logger.info("   Ready for 800+ articles/sec processing")
        else:
            logger.warning(f"⚠️ {total - successful} models failed compilation")
            for model_key, success in results.items():
                status = "✅" if success else "❌"
                logger.info(f"   {status} {model_key}")
        
        return results

def main():
    """Main compilation routine"""
    print("🚀 Scout V2 TensorRT Compiler")
    print("============================")
    
    compiler = ScoutTensorRTCompiler()
    results = compiler.compile_all_models()
    
    successful = sum(results.values())
    total = len(results)
    
    if successful == total:
        print(f"\n🎉 SUCCESS: All {total} models compiled to TensorRT!")
        print("   Ready for ultra-high performance Scout processing")
        return 0
    else:
        print(f"\n⚠️ PARTIAL SUCCESS: {successful}/{total} models compiled")
        return 1

if __name__ == "__main__":
    exit(main())