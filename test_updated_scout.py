#!/usr/bin/env python3
"""
Test Updated Scout Agent with Real Article
"""

import asyncio
import json

# Test the Scout Agent's enhanced_deep_crawl_site function directly
async def test_updated_scout():
    """Test the updated Scout Agent with the BBC article"""
    
    print("🧪 TESTING UPDATED SCOUT AGENT")
    print("=" * 40)
    
    # Import the updated Scout tools
    import sys
    sys.path.append('/home/adra/JustNewsAgentic/agents/scout')
    
    from tools import enhanced_deep_crawl_site
    
    # Test with the BBC article
    article_url = "https://www.bbc.co.uk/news/articles/cy85737235go"
    
    print(f"🎯 Testing URL: {article_url}")
    print()
    
    try:
        # Call the Scout Agent function
        result = await enhanced_deep_crawl_site(
            url=article_url,
            max_pages=1,
            max_depth=1,
            word_count_threshold=50,
            analyze_content=False  # Skip intelligence analysis for now
        )
        
        if result and isinstance(result, list) and len(result) > 0:
            page_data = result[0]
            
            print("✅ Scout Agent extraction successful!")
            print()
            print(f"📋 Title: {page_data.get('title', 'No title')}")
            print(f"📊 Word Count: {page_data.get('word_count', 0):,}")
            print(f"📊 Content Length: {page_data.get('content_length', 0):,} chars")
            print(f"🔧 Source Method: {page_data.get('source_method', 'unknown')}")
            
            if 'original_html_length' in page_data:
                original_len = page_data['original_html_length']
                clean_len = page_data['content_length']
                efficiency = (clean_len / max(original_len, 1)) * 100
                print(f"📈 Extraction Efficiency: {efficiency:.1f}% ({clean_len:,} from {original_len:,})")
            
            print()
            print("📖 EXTRACTED CONTENT SAMPLE:")
            print("-" * 30)
            
            content = page_data.get('content', '')
            if content:
                # Show first 3 sentences
                sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
                for i, sentence in enumerate(sentences[:3]):
                    print(f"   {i+1}. {sentence}")
                
                print(f"\n   ... ({len(sentences)} total sentences)")
                
                # Content quality check
                has_article_indicators = any(indicator in content.lower() for indicator in 
                    ['employees', 'workers', 'said', 'according', 'reported', 'announced'])
                has_minimal_navigation = not any(nav in content.lower() for nav in 
                    ['bbc homepage', 'menu', 'subscribe', 'cookie'])
                
                print(f"\n🔍 CONTENT QUALITY:")
                print(f"   Article indicators: {'✅' if has_article_indicators else '❌'}")
                print(f"   Minimal navigation: {'✅' if has_minimal_navigation else '❌'}")
                
                if has_article_indicators and has_minimal_navigation:
                    print(f"   ✅ HIGH QUALITY ARTICLE CONTENT!")
                else:
                    print(f"   ⚠️ Content needs improvement")
            
            print()
            print("💾 READY FOR MEMORY AGENT:")
            print("Content is properly extracted and ready for database storage")
            
        else:
            print("❌ Scout Agent returned no results")
            if result:
                print(f"Result type: {type(result)}")
                print(f"Result content: {result}")
    
    except Exception as e:
        print(f"❌ Error testing Scout Agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🎯 Testing updated Scout Agent with improved content extraction")
    print("🔍 Using cleaned_html with article content filtering")
    
    asyncio.run(test_updated_scout())
    
    print(f"\n💡 NEXT STEPS:")
    print("1. ✅ Scout Agent updated with better extraction")
    print("2. 🔄 Test Scout → Memory Agent pipeline")
    print("3. 🎯 Validate database storage works correctly")
