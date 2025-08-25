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
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        print("✅ Health check: PASSED")
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
        assert response.status_code == 200, f"Sentiment scoring failed: {response.status_code}"
        result = response.json()
        print(f"✅ Sentiment scoring: {result}")
    except Exception as e:
        print(f"❌ Sentiment scoring error: {e}")
    
    # Test bias scoring
    try:
        test_data = {
            "args": ["This is an objective news report about recent developments."],
            "kwargs": {}
        }
        response = requests.post(f"{base_url}/score_bias", json=test_data, timeout=10)
        assert response.status_code == 200, f"Bias scoring failed: {response.status_code}"
        result = response.json()
        print(f"✅ Bias scoring: {result}")
    except Exception as e:
        print(f"❌ Bias scoring error: {e}")
    
    print("\n🎉 API test completed!")

if __name__ == "__main__":
    test_agent_api()
