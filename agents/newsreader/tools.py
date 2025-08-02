"""
NewsReader Agent Tools
LLaVA-based news content extraction tools
Compatible with JustNews V4 MCP Bus system
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from llava_newsreader_agent import LlavaNewsReaderAgent

# Setup logging
logger = logging.getLogger("newsreader_tools")

# Global agent instance for efficiency
_agent_instance = None

def get_agent():
    """Get or create agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = LlavaNewsReaderAgent()
    return _agent_instance

async def extract_news_from_url(url: str, screenshot_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract news content from a URL using LLaVA vision model
    
    Args:
        url: The news article URL to process
        screenshot_path: Optional path to existing screenshot, will capture if not provided
        
    Returns:
        Dictionary containing headline, article content, and metadata
    """
    try:
        agent = get_agent()
        result = await agent.process_news_url(url, screenshot_path)
        
        return {
            "headline": result.headline,
            "article": result.article,
            "success": result.success,
            "method": result.extraction_method,
            "processing_time": result.processing_time,
            "url": url
        }
        
    except Exception as e:
        logger.error(f"News extraction failed for {url}: {str(e)}")
        return {
            "headline": "Extraction failed",
            "article": f"Error: {str(e)}",
            "success": False,
            "method": "llava-v1.5-7b",
            "processing_time": 0.0,
            "url": url
        }

async def capture_webpage_screenshot(url: str, output_path: str = "page_llava.png") -> Dict[str, Any]:
    """
    Capture a screenshot of a webpage
    
    Args:
        url: The webpage URL to screenshot
        output_path: Path where screenshot will be saved
        
    Returns:
        Dictionary with screenshot path and success status
    """
    try:
        agent = get_agent()
        screenshot_path = await agent.capture_screenshot(url, output_path)
        
        return {
            "screenshot_path": screenshot_path,
            "success": True,
            "url": url
        }
        
    except Exception as e:
        logger.error(f"Screenshot capture failed for {url}: {str(e)}")
        return {
            "screenshot_path": None,
            "success": False,
            "error": str(e),
            "url": url
        }

def analyze_image_with_llava(image_path: str) -> Dict[str, Any]:
    """
    Analyze an image using LLaVA model for news content extraction
    
    Args:
        image_path: Path to the image file to analyze
        
    Returns:
        Dictionary containing extracted headline and article content
    """
    try:
        agent = get_agent()
        result = agent.extract_content_with_llava(image_path)
        
        return {
            "headline": result["headline"],
            "article": result["article"],
            "raw_response": result.get("raw_response", ""),
            "success": True,
            "image_path": image_path
        }
        
    except Exception as e:
        logger.error(f"Image analysis failed for {image_path}: {str(e)}")
        return {
            "headline": "Analysis failed",
            "article": f"Error: {str(e)}",
            "raw_response": "",
            "success": False,
            "image_path": image_path
        }

async def extract_news_batch(urls: list, max_concurrent: int = 3) -> Dict[str, Any]:
    """
    Extract news content from multiple URLs concurrently
    
    Args:
        urls: List of news article URLs
        max_concurrent: Maximum number of concurrent extractions
        
    Returns:
        Dictionary with results for each URL
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def extract_single(url):
        async with semaphore:
            return await extract_news_from_url(url)
    
    try:
        tasks = [extract_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        output = {
            "results": {},
            "success_count": 0,
            "total_count": len(urls),
            "errors": []
        }
        
        for i, result in enumerate(results):
            url = urls[i]
            if isinstance(result, Exception):
                output["results"][url] = {
                    "success": False,
                    "error": str(result)
                }
                output["errors"].append(f"{url}: {str(result)}")
            else:
                output["results"][url] = result
                if result.get("success", False):
                    output["success_count"] += 1
        
        return output
        
    except Exception as e:
        logger.error(f"Batch extraction failed: {str(e)}")
        return {
            "results": {},
            "success_count": 0,
            "total_count": len(urls),
            "errors": [f"Batch operation failed: {str(e)}"]
        }

# Convenience functions for testing
async def test_news_extraction():
    """Test function for development"""
    test_url = "https://www.bbc.co.uk/news/uk-politics-55497671"
    
    print("🧪 Testing LLaVA NewsReader Tools")
    print("="*50)
    
    # Test single extraction
    result = await extract_news_from_url(test_url)
    
    print(f"URL: {test_url}")
    print(f"Success: {result['success']}")
    print(f"Method: {result['method']}")
    print(f"Processing Time: {result['processing_time']:.2f}s")
    print(f"Headline: {result['headline']}")
    print(f"Article: {result['article'][:200]}...")
    print("="*50)
    
    return result

if __name__ == "__main__":
    # Direct testing
    asyncio.run(test_news_extraction())
