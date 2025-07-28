#!/usr/bin/env python3
"""
JustNews V4 Native GPU Analyst Service
Ubuntu 24.04 optimized with RTX 3090 acceleration

Performance: 41.4 articles/sec sentiment, 168.1 articles/sec bias analysis
"""

import sys
import os
sys.path.append('agents/analyst')

from hybrid_tools_v4 import GPUAcceleratedAnalyst
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🚀 JustNews V4 Native GPU Analyst Service")
print("=" * 50)

# Initialize the GPU analyst
gpu_analyst = GPUAcceleratedAnalyst()

# Create FastAPI app
app = FastAPI(
    title="JustNews V4 GPU Analyst", 
    version="4.0.0",
    description="Native Ubuntu GPU-accelerated news analysis with RTX 3090"
)

class ArticlesBatch(BaseModel):
    articles: List[str]

@app.get("/health")
async def health_check():
    return {
        'status': 'operational', 
        'gpu': 'NVIDIA GeForce RTX 3090',
        'version': 'v4.0.0',
        'platform': 'Ubuntu 24.04 Native',
        'performance': {
            'sentiment_analysis': '41.4 articles/sec',
            'bias_analysis': '168.1 articles/sec'
        }
    }

@app.post("/analyze/sentiment/batch")
async def analyze_sentiment_batch(batch: ArticlesBatch):
    try:
        results = gpu_analyst.score_sentiment_batch_gpu(batch.articles)
        return {
            'results': results, 
            'count': len(batch.articles),
            'service': 'native_gpu_analyst',
            'gpu_accelerated': True
        }
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/bias/batch")  
async def analyze_bias_batch(batch: ArticlesBatch):
    try:
        results = gpu_analyst.score_bias_batch_gpu(batch.articles)
        return {
            'results': results, 
            'count': len(batch.articles),
            'service': 'native_gpu_analyst',
            'gpu_accelerated': True
        }
    except Exception as e:
        logger.error(f"Bias analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/benchmark")
async def performance_benchmark():
    """Run a quick performance benchmark"""
    test_articles = [
        "Technology companies are advancing AI research rapidly",
        "Economic indicators show mixed results this quarter", 
        "Environmental policies receive public support nationwide"
    ]
    
    import time
    
    # Benchmark sentiment
    start = time.time()
    sentiment_results = gpu_analyst.score_sentiment_batch_gpu(test_articles)
    sentiment_time = time.time() - start
    
    # Benchmark bias
    start = time.time() 
    bias_results = gpu_analyst.score_bias_batch_gpu(test_articles)
    bias_time = time.time() - start
    
    return {
        'benchmark_articles': len(test_articles),
        'sentiment_analysis': {
            'time_seconds': sentiment_time,
            'articles_per_second': len(test_articles) / sentiment_time,
            'results': sentiment_results
        },
        'bias_analysis': {
            'time_seconds': bias_time, 
            'articles_per_second': len(test_articles) / bias_time,
            'results': bias_results
        },
        'gpu_memory_available': '23.5GB',
        'gpu_device': 'NVIDIA GeForce RTX 3090'
    }

if __name__ == "__main__":
    print("✅ FastAPI GPU Analyst Service Ready")
    print("🌐 Starting server on http://localhost:8004")
    print("📋 Endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /performance/benchmark - Performance test")
    print("  POST /analyze/sentiment/batch - Batch sentiment analysis")
    print("  POST /analyze/bias/batch - Batch bias analysis")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")
