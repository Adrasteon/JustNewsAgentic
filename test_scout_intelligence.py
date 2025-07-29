#!/usr/bin/env python3
"""
Test Scout intelligence with static content
"""

import requests
import json

SCOUT_URL = "http://localhost:8002"

def test_scout_with_static_content():
    """Test Scout with predefined content"""
    print("🧠 Testing Scout intelligence with static content...")
    
    # Test with news-like content
    news_content = {
        "url": "https://example-news.com/tech-breakthrough",
        "content": """
        # Breaking: Revolutionary AI Breakthrough Announced
        
        Scientists at MIT have announced a groundbreaking development in artificial intelligence 
        that could revolutionize how we understand machine learning. The new algorithm shows 
        unprecedented capabilities in natural language processing and reasoning.
        
        The research team, led by Dr. Sarah Johnson, published their findings in Nature AI today.
        "This represents a fundamental shift in how AI systems can understand and generate human-like text," 
        Johnson explained in an interview.
        
        Key findings include:
        - 40% improvement in language understanding
        - Reduced computational requirements by 60%
        - Enhanced bias detection and mitigation
        
        The technology is expected to be commercially available within the next two years,
        with potential applications in healthcare, education, and scientific research.
        """,
        "query": "AI technology breakthrough news"
    }
    
    # Test with non-news content
    non_news_content = {
        "url": "https://example-recipe.com/chocolate-cake",
        "content": """
        # Best Chocolate Cake Recipe
        
        This delicious chocolate cake recipe is perfect for any occasion. 
        
        Ingredients:
        - 2 cups flour
        - 1 cup sugar
        - 1/2 cup cocoa powder
        - 2 eggs
        - 1 cup milk
        
        Instructions:
        1. Preheat oven to 350°F
        2. Mix dry ingredients
        3. Add wet ingredients
        4. Bake for 30 minutes
        
        Enjoy your homemade chocolate cake!
        """,
        "query": "chocolate cake recipe"
    }
    
    # Test both contents
    test_cases = [
        ("News Content", news_content),
        ("Recipe Content", non_news_content)
    ]
    
    for test_name, content in test_cases:
        print(f"\n📰 Testing {test_name}...")
        try:
            response = requests.post(
                f"{SCOUT_URL}/intelligent_content_crawl",
                json={"args": [content], "kwargs": {}},
                timeout=30
            )
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Analysis complete!")
                
                # Display key results
                if isinstance(result, dict):
                    print(f"URL: {result.get('url', 'N/A')}")
                    print(f"Scout Score: {result.get('scout_score', 'N/A')}")
                    print(f"Is News: {result.get('is_news', 'N/A')}")
                    print(f"Recommendation: {result.get('recommendation', 'N/A')}")
                    
                    if result.get('quality_metrics'):
                        print(f"Quality Metrics: {result['quality_metrics']}")
                    
                    if result.get('scout_analysis'):
                        analysis = result['scout_analysis']
                        print(f"Classification: {analysis.get('news_classification', {})}")
                        print(f"Quality Assessment: {analysis.get('quality_assessment', {})}")
                        
                else:
                    print(f"Result: {result}")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")

def main():
    print("🚀 Testing Scout Agent Intelligence")
    print("=" * 50)
    
    # Check if Scout is running
    try:
        response = requests.get(f"{SCOUT_URL}/health", timeout=5)
        print(f"✅ Scout Agent: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Scout Agent not available: {e}")
        return
    
    test_scout_with_static_content()
    
    print("\n" + "=" * 50)
    print("🏁 Test complete!")

if __name__ == "__main__":
    main()
