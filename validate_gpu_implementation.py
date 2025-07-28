#!/usr/bin/env python3
"""
Multi-Agent GPU Implementation Validation
Validates all GPU implementations without external dependencies
"""

import os
import sys
import importlib.util
from typing import Dict, List, Any

def validate_implementation_exists(file_path: str, description: str) -> Dict[str, Any]:
    """Validate that implementation file exists and has correct structure"""
    result = {
        "name": description,
        "path": file_path,
        "exists": False,
        "imports": False,
        "key_classes": [],
        "key_functions": [],
        "issues": []
    }
    
    try:
        # Check file exists
        if not os.path.exists(file_path):
            result["issues"].append("File not found")
            return result
        result["exists"] = True
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key patterns
        if "class " in content:
            classes = [line.strip().split('class ')[1].split('(')[0].split(':')[0] 
                      for line in content.split('\n') if line.strip().startswith('class ')]
            result["key_classes"] = classes
            
        if "def " in content:
            functions = [line.strip().split('def ')[1].split('(')[0] 
                        for line in content.split('\n') if line.strip().startswith('def ')]
            result["key_functions"] = functions[:10]  # Limit to first 10
            
        # Check for GPU-specific patterns
        gpu_patterns = ["torch", "cuda", "gpu", "GPU", "batch_size", "device"]
        gpu_found = sum(1 for pattern in gpu_patterns if pattern in content)
        result["gpu_patterns"] = gpu_found
        
        # Check import section (basic validation)
        import_section = content.split('\n')[:50]  # First 50 lines usually contain imports
        try_import_found = any("try:" in line and ("import" in content[content.find(line):content.find(line)+200]) 
                              for line in import_section)
        result["has_safe_imports"] = try_import_found
        
        print(f"✅ {description}")
        print(f"   Classes: {', '.join(result['key_classes'][:3])}")
        print(f"   GPU patterns: {gpu_found}/6")
        print(f"   Safe imports: {try_import_found}")
        
    except Exception as e:
        result["issues"].append(f"Validation error: {str(e)}")
        print(f"❌ {description}: {str(e)}")
    
    return result

def validate_api_integration(agent_path: str, agent_name: str) -> Dict[str, Any]:
    """Validate that agent main.py has GPU endpoints"""
    main_file = os.path.join(agent_path, "main.py")
    result = {
        "agent": agent_name,
        "main_exists": False,
        "gpu_endpoints": [],
        "issues": []
    }
    
    try:
        if not os.path.exists(main_file):
            result["issues"].append("main.py not found")
            return result
        result["main_exists"] = True
        
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for GPU endpoints
        lines = content.split('\n')
        gpu_endpoints = []
        for i, line in enumerate(lines):
            if "@app.post(" in line and "_gpu" in line:
                endpoint_name = line.split('"')[1] if '"' in line else "unknown"
                gpu_endpoints.append(endpoint_name)
        
        result["gpu_endpoints"] = gpu_endpoints
        
        # Check for GPU tool imports
        has_gpu_import = "from gpu_tools import" in content
        result["has_gpu_import"] = has_gpu_import
        
        print(f"✅ {agent_name} API Integration")
        print(f"   GPU endpoints: {', '.join(gpu_endpoints)}")
        print(f"   GPU imports: {has_gpu_import}")
        
    except Exception as e:
        result["issues"].append(f"API validation error: {str(e)}")
        print(f"❌ {agent_name} API: {str(e)}")
    
    return result

def main():
    """Main validation runner"""
    print("🚀 Multi-Agent GPU Implementation Validation")
    print("=" * 60)
    
    base_path = "/home/adra/JustNewsAgentic"
    
    # Validate core implementations
    implementations = [
        (os.path.join(base_path, "agents/common/gpu_manager.py"), "Multi-Agent GPU Manager"),
        (os.path.join(base_path, "agents/fact_checker/gpu_tools.py"), "Fact Checker GPU Tools"),
        (os.path.join(base_path, "agents/synthesizer/gpu_tools.py"), "Synthesizer GPU Tools"),
        (os.path.join(base_path, "agents/critic/gpu_tools.py"), "Critic GPU Tools"),
    ]
    
    implementation_results = []
    for file_path, description in implementations:
        result = validate_implementation_exists(file_path, description)
        implementation_results.append(result)
        print()
    
    # Validate API integrations
    agents = [
        ("agents/fact_checker", "Fact Checker"),
        ("agents/synthesizer", "Synthesizer"), 
        ("agents/critic", "Critic"),
    ]
    
    api_results = []
    for agent_path, agent_name in agents:
        full_path = os.path.join(base_path, agent_path)
        result = validate_api_integration(full_path, agent_name)
        api_results.append(result)
        print()
    
    # Validate test files
    test_files = [
        (os.path.join(base_path, "test_gpu_manager.py"), "GPU Manager Tests"),
        (os.path.join(base_path, "test_fact_checker_gpu.py"), "Fact Checker GPU Tests"),
        (os.path.join(base_path, "test_synthesizer_gpu.py"), "Synthesizer GPU Tests"),
    ]
    
    test_results = []
    for file_path, description in test_files:
        result = validate_implementation_exists(file_path, description)
        test_results.append(result)
        print()
    
    # Summary Report
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    # Implementation Summary
    impl_success = sum(1 for r in implementation_results if r["exists"] and not r["issues"])
    print(f"🔧 Core Implementations: {impl_success}/{len(implementation_results)} ✅")
    
    # API Integration Summary
    api_success = sum(1 for r in api_results if r["main_exists"] and r["gpu_endpoints"])
    print(f"🌐 API Integrations: {api_success}/{len(api_results)} ✅")
    
    # Test Coverage Summary
    test_success = sum(1 for r in test_results if r["exists"] and not r["issues"])
    print(f"🧪 Test Coverage: {test_success}/{len(test_results)} ✅")
    
    # Overall Status
    total_components = len(implementation_results) + len(api_results) + len(test_results)
    total_success = impl_success + api_success + test_success
    success_rate = (total_success / total_components) * 100
    
    print(f"\n🎯 Overall Implementation: {total_success}/{total_components} ({success_rate:.0f}%)")
    
    if success_rate >= 90:
        print("✅ READY FOR PRODUCTION TESTING")
        print("   Next: Deploy with 'docker-compose up --build'")
    elif success_rate >= 70:
        print("⚠️ MINOR ISSUES - READY FOR VALIDATION")
        print("   Next: Fix minor issues then deploy")
    else:
        print("❌ MAJOR ISSUES - REQUIRES ATTENTION")
        print("   Next: Address implementation issues")
    
    # Detailed issue report
    all_issues = []
    for results in [implementation_results, api_results, test_results]:
        for result in results:
            if result.get("issues"):
                all_issues.extend([(result["name"], issue) for issue in result["issues"]])
    
    if all_issues:
        print(f"\n⚠️ Issues Found ({len(all_issues)}):")
        for name, issue in all_issues[:5]:  # Show first 5 issues
            print(f"   • {name}: {issue}")
        if len(all_issues) > 5:
            print(f"   ... and {len(all_issues) - 5} more")
    
    return success_rate >= 90

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
