#!/usr/bin/env python3
"""
Simple Article Pipeline Test
Test basic functionality of key agents
"""

import requests
import json
import time
from datetime import datetime

# Configuration
MCP_BUS_URL = "http://localhost:8000"

def test_agent_health():
    """Test all agent health endpoints"""
    print("🔍 Testing Agent Health")
    print("=" * 30)
    
    agents = {
        "MCP Bus": 8000,
        "Chief Editor": 8001, 
        "Scout": 8002,
        "Fact Checker": 8003,
        "Analyst": 8004,
        "Synthesizer": 8005,
        "Critic": 8006,
        "Memory": 8007,
        "Reasoning": 8008,
        "NewsReader": 8009
    }
    
    healthy_agents = 0
    for name, port in agents.items():
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: HEALTHY")
                healthy_agents += 1
            else:
                print(f"❌ {name}: UNHEALTHY ({response.status_code})")
        except Exception as e:
            print(f"❌ {name}: ERROR ({str(e)[:50]})")
    
    print(f"\n📊 Health Summary: {healthy_agents}/{len(agents)} agents healthy")
    return healthy_agents == len(agents)

def test_mcp_bus_registrations():
    """Test MCP Bus agent registrations"""
    print("\n🔗 Testing MCP Bus Registrations")
    print("=" * 35)
    
    try:
        response = requests.get(f"{MCP_BUS_URL}/agents", timeout=5)
        if response.status_code == 200:
            agents = response.json()
            print(f"✅ Registered agents: {len(agents)}")
            for name, url in agents.items():
                print(f"   - {name}: {url}")
            return True
        else:
            print(f"❌ MCP Bus registration check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MCP Bus error: {e}")
        return False

def test_memory_storage():
    """Test Memory agent storage with simple data"""
    print("\n💾 Testing Memory Storage")
    print("=" * 25)
    
    test_article = {
        "content": "This is a test article for pipeline validation",
        "metadata": {
            "source": "pipeline_test",
            "timestamp": datetime.now().isoformat(),
            "test": True
        }
    }
    
    try:
        # Test direct Memory agent call
        response = requests.post(
            "http://localhost:8007/save_article",
            json={"args": [], "kwargs": test_article},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Memory storage: SUCCESS")
            print(f"   Result: {result}")
            return True
        else:
            print(f"❌ Memory storage failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Memory storage error: {e}")
        return False

def test_analyst_simple():
    """Test Analyst with simple text"""
    print("\n📊 Testing Analyst Agent")
    print("=" * 25)
    
    test_text = "This is a positive news article about technological advancement."
    
    try:
        # Test direct Analyst agent call
        response = requests.post(
            "http://localhost:8004/analyze_sentiment",
            json={"args": [test_text], "kwargs": {}},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analyst processing: SUCCESS")
            print(f"   Result: {result}")
            return True
        else:
            print(f"❌ Analyst processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Analyst processing error: {e}")
        return False

def test_reasoning_simple():
    """Test Reasoning agent with simple facts"""
    print("\n🧠 Testing Reasoning Agent")
    print("=" * 26)
    
    try:
        # Test adding a simple fact
        response = requests.post(
            "http://localhost:8008/add_facts",
            json={"args": [["test_fact = true", "pipeline_test = active"]], "kwargs": {}},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Reasoning add facts: SUCCESS")
            print(f"   Result: {result}")
            
            # Test querying the fact
            query_response = requests.post(
                "http://localhost:8008/query",
                json={"args": ["test_fact"], "kwargs": {}},
                timeout=10
            )
            
            if query_response.status_code == 200:
                query_result = query_response.json()
                print(f"✅ Reasoning query: SUCCESS")
                print(f"   Query result: {query_result}")
                return True
            else:
                print(f"❌ Reasoning query failed: {query_response.status_code}")
                return False
        else:
            print(f"❌ Reasoning add facts failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Reasoning processing error: {e}")
        return False

def main():
    """Run simple pipeline tests"""
    print("🚀 JustNews V4 Simple Pipeline Test")
    print("=" * 40)
    print(f"Test started at: {datetime.now().isoformat()}")
    print()
    
    tests = [
        ("Agent Health", test_agent_health),
        ("MCP Bus Registrations", test_mcp_bus_registrations),
        ("Memory Storage", test_memory_storage),
        ("Analyst Processing", test_analyst_simple),
        ("Reasoning Engine", test_reasoning_simple)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 PIPELINE TEST SUMMARY")
    print("=" * 40)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - System is operational!")
    else:
        print("⚠️ Some tests failed - Check individual components")
    
    print(f"Test completed at: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
