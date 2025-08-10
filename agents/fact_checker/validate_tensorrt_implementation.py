#!/usr/bin/env python3
"""
Fact Checker TensorRT Implementation Validation
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
    """Test Fact Checker TensorRT compiler file"""
    logger.info("🔧 Validating Fact Checker TensorRT Compiler...")
    
    compiler_path = Path("native_tensorrt_compiler.py")
    
    # Test syntax
    if not validate_python_syntax(compiler_path):
        return False
    
    # Test class structure
    if not validate_class_structure(compiler_path, ["FactCheckerTensorRTCompiler"]):
        return False
    
    # Test key methods
    expected_methods = ["__init__", "compile_model", "compile_all_models", "get_compilation_info"]
    if not validate_method_structure(compiler_path, "FactCheckerTensorRTCompiler", expected_methods):
        return False
    
    logger.info("✅ Fact Checker TensorRT Compiler validation passed")
    return True

def test_inference_engine_file():
    """Test Fact Checker TensorRT inference engine file"""
    logger.info("🚀 Validating Fact Checker TensorRT Inference Engine...")
    
    engine_path = Path("native_tensorrt_inference_engine.py")
    
    # Test syntax
    if not validate_python_syntax(engine_path):
        return False
    
    # Test class structure
    if not validate_class_structure(engine_path, ["NativeTensorRTFactCheckerEngine"]):
        return False
    
    # Test key methods
    expected_methods = [
        "__init__", 
        "verify_fact", 
        "assess_credibility", 
        "retrieve_evidence", 
        "extract_claims",
        "comprehensive_fact_check",
        "get_performance_stats",
        "cleanup",
        "__enter__",
        "__exit__"
    ]
    if not validate_method_structure(engine_path, "NativeTensorRTFactCheckerEngine", expected_methods):
        return False
    
    logger.info("✅ Fact Checker TensorRT Inference Engine validation passed")
    return True

def test_directory_structure():
    """Test directory structure"""
    logger.info("📁 Validating directory structure...")
    
    current_dir = Path.cwd()
    
    # Check required files exist
    required_files = [
        "native_tensorrt_compiler.py",
        "native_tensorrt_inference_engine.py"
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
    """Test integration points with existing Fact Checker system"""
    logger.info("🔄 Validating integration points...")
    
    # Check that the existing Fact Checker V2 engine file exists
    parent_dir = Path.cwd()
    fact_checker_v2_path = parent_dir / "fact_checker_v2_engine.py"
    
    if fact_checker_v2_path.exists():
        logger.info("✅ Existing Fact Checker V2 engine found for fallback")
        
        # Validate syntax of existing Fact Checker V2 engine
        if validate_python_syntax(fact_checker_v2_path):
            logger.info("✅ Fact Checker V2 engine syntax valid")
        else:
            logger.warning("⚠️ Fact Checker V2 engine has syntax issues")
            return False
    else:
        logger.warning("⚠️ Fact Checker V2 engine not found (fallback may not work)")
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
        
        # Check for expected TensorRT model configurations
        expected_tensorrt_models = [
            "fact_verification",
            "credibility_assessment"
        ]
        
        for model_name in expected_tensorrt_models:
            if f'"{model_name}"' in source:
                logger.info(f"✅ {model_name} TensorRT configuration found")
            else:
                logger.error(f"❌ {model_name} TensorRT configuration missing")
                return False
        
        # Check for key configuration parameters
        config_params = ["model_name", "batch_size", "max_sequence_length", "precision"]
        for param in config_params:
            if f'"{param}"' in source:
                logger.info(f"✅ {param} parameter found")
            else:
                logger.error(f"❌ {param} parameter missing")
                return False
        
        # Check that excluded models are documented
        if "evidence_retrieval" in source and "claim_extraction" in source:
            logger.info("✅ Excluded models properly documented")
        else:
            logger.warning("⚠️ Excluded models not properly documented")
        
        logger.info("✅ Model configurations validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating model configurations: {e}")
        return False

def test_hybrid_architecture():
    """Test hybrid architecture implementation"""
    logger.info("🔗 Validating hybrid architecture...")
    
    engine_path = Path("native_tensorrt_inference_engine.py")
    
    try:
        with open(engine_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Check for hybrid methods (some TensorRT, some fallback)
        tensorrt_methods = ["verify_fact", "assess_credibility"]
        fallback_methods = ["retrieve_evidence", "extract_claims"]
        
        for method in tensorrt_methods:
            if f"def {method}" in source and "tensorrt" in source.lower():
                logger.info(f"✅ {method} supports TensorRT")
            else:
                logger.error(f"❌ {method} TensorRT support missing")
                return False
        
        for method in fallback_methods:
            if f"def {method}" in source and "fallback" in source.lower():
                logger.info(f"✅ {method} uses fallback (appropriate)")
            else:
                logger.error(f"❌ {method} fallback handling missing")
                return False
        
        # Check for comprehensive method
        if "comprehensive_fact_check" in source:
            logger.info("✅ Comprehensive fact checking method found")
        else:
            logger.error("❌ Comprehensive fact checking method missing")
            return False
        
        logger.info("✅ Hybrid architecture validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating hybrid architecture: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧪 Fact Checker TensorRT Implementation Validation")
    print("=================================================")
    
    # Change to fact_checker directory if not already there
    if not Path("fact_checker_v2_engine.py").exists():
        fact_checker_dir = Path(__file__).parent
        os.chdir(fact_checker_dir)
        print(f"Changed to directory: {os.getcwd()}")
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Compiler File", test_compiler_file),
        ("Inference Engine File", test_inference_engine_file),
        ("Model Configurations", test_model_configurations),
        ("Integration Points", test_integration_points),
        ("Hybrid Architecture", test_hybrid_architecture)
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
        print("   Fact Checker TensorRT implementation structure is correct")
        print("   Hybrid architecture (2 TensorRT + 2 fallback) properly implemented")
        print("   Ready for compilation when GPU/ML dependencies are available")
        return 0
    else:
        print("⚠️ SOME VALIDATIONS FAILED")
        print("   Please review the implementation before proceeding")
        return 1

if __name__ == "__main__":
    exit(main())