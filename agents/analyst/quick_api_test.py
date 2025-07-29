#!/usr/bin/env python3
"""
Quick API test for the running analyst agent
"""

import requests
import json

def test_agent_api():
    """Test the analyst agent API endpoints"""
    base_url = "http://localhost:8004"
    
    print("🚀 Testing Native TensorRT Analyst Agent API")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check: PASSED")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test sentiment scoring
    try:
        test_data = {
            "args": ["This is a great news article about technological innovation!"],
            "kwargs": {}
        }
        response = requests.post(f"{base_url}/score_sentiment", json=test_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Sentiment scoring: {result}")
        else:
            print(f"❌ Sentiment scoring failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Sentiment scoring error: {e}")
    
    # Test bias scoring
    try:
        test_data = {
            "args": ["This is an objective news report about recent developments."],
            "kwargs": {}
        }
        response = requests.post(f"{base_url}/score_bias", json=test_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Bias scoring: {result}")
        else:
            print(f"❌ Bias scoring failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Bias scoring error: {e}")
    
    print("\n🎉 API test completed!")
    return True

if __name__ == "__main__":
    test_agent_api()
