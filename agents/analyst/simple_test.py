#!/usr/bin/env python3
"""
Simple test of native TensorRT analyst functions
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_direct_functions():
    """Test the native TensorRT functions directly"""
    print("🚀 Testing Native TensorRT Functions Directly")
    print("=" * 50)
    
    try:
        # Import the functions
        from tensorrt_tools import score_sentiment, score_bias
        logger.info("✅ Successfully imported TensorRT tools")
        
        # Test sentiment scoring
        test_text = "This is a great news article about technological advancement!"
        sentiment_score = score_sentiment(test_text)
        logger.info(f"✅ Sentiment Score: {sentiment_score}")
        
        # Test bias scoring  
        bias_score = score_bias(test_text)
        logger.info(f"✅ Bias Score: {bias_score}")
        
        print("\n🎉 All direct function tests passed!")
        print(f"📊 Results: Sentiment={sentiment_score:.3f}, Bias={bias_score:.3f}")
        # Pytest expects tests to assert rather than return values
        assert isinstance(sentiment_score, float)
        assert isinstance(bias_score, float)
        assert 0.0 <= sentiment_score <= 1.0
        assert 0.0 <= bias_score <= 1.0

    except Exception as e:
        logger.error(f"❌ Direct function test failed: {e}")
        # Re-raise so pytest records the failure
        raise

if __name__ == "__main__":
    success = test_direct_functions()
    sys.exit(0 if success else 1)
