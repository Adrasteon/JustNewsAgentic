#!/usr/bin/env python3
"""
Scout TensorRT Implementation Validation
Tests code structure and architecture without requiring ML dependencies

This validates the implementation design and integration points
"""

import sys
import os
from pathlib import Path
import ast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_python_syntax(file_path: Path) -> bool:
    """Validate Python syntax of a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True
    except SyntaxError as e:
        logger.error(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error parsing {file_path}: {e}")
        return False

def validate_class_structure(file_path: Path, expected_classes: list) -> bool:
    """Validate that expected classes exist in file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        found_classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                found_classes.append(node.name)
        
        for expected_class in expected_classes:
            if expected_class in found_classes:
                logger.info(f"✅ Class {expected_class} found")
            else:
                logger.error(f"❌ Class {expected_class} missing")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating class structure: {e}")
        return False

def validate_method_structure(file_path: Path, class_name: str, expected_methods: list) -> bool:
    """Validate that expected methods exist in a class"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        found_methods = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        found_methods.append(item.name)
                break
        
        for expected_method in expected_methods:
            if expected_method in found_methods:
                logger.info(f"✅ Method {class_name}.{expected_method} found")
            else:
                logger.error(f"❌ Method {class_name}.{expected_method} missing")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating method structure: {e}")
        return False

def test_compiler_file():
    """Test Scout TensorRT compiler file"""
    logger.info("🔧 Validating Scout TensorRT Compiler...")
    
    compiler_path = Path("native_tensorrt_compiler.py")
    
    # Test syntax
    if not validate_python_syntax(compiler_path):
        return False
    
    # Test class structure
    if not validate_class_structure(compiler_path, ["ScoutTensorRTCompiler"]):
        return False
    
    # Test key methods
    expected_methods = ["__init__", "compile_model", "compile_all_models"]
    if not validate_method_structure(compiler_path, "ScoutTensorRTCompiler", expected_methods):
        return False
    
    logger.info("✅ Scout TensorRT Compiler validation passed")
    return True

def test_inference_engine_file():
    """Test Scout TensorRT inference engine file"""
    logger.info("🚀 Validating Scout TensorRT Inference Engine...")
    
    engine_path = Path("native_tensorrt_inference_engine.py")
    
    # Test syntax
    if not validate_python_syntax(engine_path):
        return False
    
    # Test class structure
    if not validate_class_structure(engine_path, ["NativeTensorRTScoutEngine"]):
        return False
    
    # Test key methods
    expected_methods = [
        "__init__", 
        "classify_news", 
        "assess_quality", 
        "analyze_sentiment", 
        "detect_bias",
        "get_performance_stats",
        "cleanup",
        "__enter__",
        "__exit__"
    ]
    if not validate_method_structure(engine_path, "NativeTensorRTScoutEngine", expected_methods):
        return False
    
    logger.info("✅ Scout TensorRT Inference Engine validation passed")
    return True

def test_directory_structure():
    """Test directory structure"""
    logger.info("📁 Validating directory structure...")
    
    current_dir = Path.cwd()
    
    # Check required files exist
    required_files = [
        "native_tensorrt_compiler.py",
        "native_tensorrt_inference_engine.py",
        "test_tensorrt_implementation.py"
    ]
    
    for file_name in required_files:
        file_path = current_dir / file_name
        if file_path.exists():
            logger.info(f"✅ {file_name} exists")
        else:
            logger.error(f"❌ {file_name} missing")
            return False
    
    # Check TensorRT engines directory
    engines_dir = current_dir / "tensorrt_engines"
    if engines_dir.exists() and engines_dir.is_dir():
        logger.info("✅ tensorrt_engines directory exists")
    else:
        logger.error("❌ tensorrt_engines directory missing")
        return False
    
    logger.info("✅ Directory structure validation passed")
    return True

def test_integration_points():
    """Test integration points with existing Scout system"""
    logger.info("🔄 Validating integration points...")
    
    # Check that the existing Scout V2 engine file exists
    parent_dir = Path.cwd()
    scout_v2_path = parent_dir / "gpu_scout_engine_v2.py"
    
    if scout_v2_path.exists():
        logger.info("✅ Existing Scout V2 engine found for fallback")
        
        # Validate syntax of existing Scout V2 engine
        if validate_python_syntax(scout_v2_path):
            logger.info("✅ Scout V2 engine syntax valid")
        else:
            logger.warning("⚠️ Scout V2 engine has syntax issues")
            return False
    else:
        logger.warning("⚠️ Scout V2 engine not found (fallback may not work)")
        return False
    
    logger.info("✅ Integration points validation passed")
    return True

def test_model_configurations():
    """Test model configurations in the compiler"""
    logger.info("⚙️ Validating model configurations...")
    
    compiler_path = Path("native_tensorrt_compiler.py")
    
    try:
        with open(compiler_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Check for expected model configurations
        expected_models = [
            "news_classifier",
            "quality_assessor", 
            "sentiment_analyzer",
            "bias_detector"
        ]
        
        for model_name in expected_models:
            if f'"{model_name}"' in source:
                logger.info(f"✅ {model_name} configuration found")
            else:
                logger.error(f"❌ {model_name} configuration missing")
                return False
        
        # Check for key configuration parameters
        config_params = ["model_name", "batch_size", "max_sequence_length", "precision"]
        for param in config_params:
            if f'"{param}"' in source:
                logger.info(f"✅ {param} parameter found")
            else:
                logger.error(f"❌ {param} parameter missing")
                return False
        
        logger.info("✅ Model configurations validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating model configurations: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧪 Scout TensorRT Implementation Validation")
    print("==========================================")
    
    # Change to scout directory if not already there
    if not Path("gpu_scout_engine_v2.py").exists():
        scout_dir = Path(__file__).parent
        os.chdir(scout_dir)
        print(f"Changed to directory: {os.getcwd()}")
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Compiler File", test_compiler_file),
        ("Inference Engine File", test_inference_engine_file),
        ("Model Configurations", test_model_configurations),
        ("Integration Points", test_integration_points)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n=== {test_name} ===")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 VALIDATION RESULTS")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("   Scout TensorRT implementation structure is correct")
        print("   Ready for compilation when GPU/ML dependencies are available")
        return 0
    else:
        print("⚠️ SOME VALIDATIONS FAILED")
        print("   Please review the implementation before proceeding")
        return 1

if __name__ == "__main__":
    exit(main())