#!/usr/bin/env python3
"""
Direct test of Memory Agent to debug 422 error
"""

import requests
import json

def test_memory_agent_direct():
    """Test Memory Agent endpoints directly"""
    
    print("🔍 DIRECT MEMORY AGENT TEST")
    print("=" * 30)
    
    # Test the exact payload format that should work
    test_payload = {
        "args": [{
            "content": "This is a test article about a skyscraper incident in New York.",
            "metadata": {
                "url": "https://test.com/article",
                "title": "Test Article",
                "source": "Test Source",
                "word_count": 12,
                "content_length": 63,
                "extraction_method": "test",
                "test_pipeline": True
            }
        }],
        "kwargs": {}
    }
    
    print("🧪 Testing save_article endpoint...")
    print(f"Payload: {json.dumps(test_payload, indent=2)}")
    
    try:
        response = requests.post(
            "http://localhost:8007/save_article",
            json=test_payload,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Direct Memory Agent call successful!")
            return True
        else:
            print("❌ Direct Memory Agent call failed")
            
            # Try alternative format
            print("\n🔄 Trying direct format (no args/kwargs wrapper)...")
            direct_payload = test_payload["args"][0]
            
            response2 = requests.post(
                "http://localhost:8007/save_article",
                json=direct_payload,
                timeout=10
            )
            
            print(f"Direct Status Code: {response2.status_code}")
            print(f"Direct Response: {response2.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False

if __name__ == "__main__":
    test_memory_agent_direct()
