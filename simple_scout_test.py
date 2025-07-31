#!/usr/bin/env python3
"""
Simple Scout Agent Test
Just test the Scout Agent response format
"""

import requests
import json

# Test Scout Agent directly
print("Testing Scout Agent directly...")
try:
    response = requests.post("http://localhost:8002/enhanced_deep_crawl_site", 
                           json={"args": ["https://www.bbc.com/news"], 
                                "kwargs": {"max_depth": 1, "max_pages": 5}}, 
                           timeout=30)
    print(f"Direct Scout Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Direct Scout Response Type: {type(data)}")
        print(f"Direct Scout Response Sample: {str(data)[:300]}...")
except Exception as e:
    print(f"Direct Scout Error: {e}")

print("\nTesting Scout Agent via MCP Bus...")
try:
    response = requests.post("http://localhost:8000/call", 
                           json={"agent": "scout", 
                                "tool": "enhanced_deep_crawl_site",
                                "args": ["https://www.bbc.com/news"],
                                "kwargs": {"max_depth": 1, "max_pages": 5}}, 
                           timeout=30)
    print(f"MCP Bus Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"MCP Bus Response Type: {type(data)}")
        print(f"MCP Bus Response Sample: {str(data)[:300]}...")
except Exception as e:
    print(f"MCP Bus Error: {e}")
