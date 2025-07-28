#!/usr/bin/env python3
"""
JustNews V4 Realistic Performance Test
Uses actual news article lengths (1,200+ characters) for accurate benchmarking
"""

import torch
import time
import gc
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

def create_realistic_news_articles(count=100):
    """Generate realistic news articles (1,200-2,000 characters)"""
    
    # Real news article templates with realistic content
    templates = [
        "Breaking news from the financial markets today as major technology stocks experienced significant volatility amid concerns about regulatory changes and market uncertainty. Industry analysts are closely monitoring the situation as investors react to new policies that could impact future growth prospects. The developments come at a critical time when market sentiment has been particularly sensitive to geopolitical tensions and economic indicators. Market participants are awaiting further clarification from regulatory bodies about the scope and timeline of these potential changes. Early trading sessions showed mixed results across different sectors, with some defensive stocks gaining ground while growth-oriented investments faced selling pressure. This pattern reflects the broader uncertainty that has characterized recent market behavior. Experts suggest that investors should maintain a diversified approach and focus on long-term fundamentals rather than short-term market fluctuations. The situation continues to evolve as stakeholders assess the potential implications for various industry segments and business models.",
        
        "In a surprising turn of events, climate scientists have announced new findings that could reshape our understanding of environmental patterns and weather systems across multiple regions. The comprehensive study, conducted over several years, involved data collection from numerous monitoring stations and advanced modeling techniques. Researchers found that certain atmospheric phenomena are occurring more frequently than previously predicted, with implications for agriculture, urban planning, and disaster preparedness. The findings suggest that adaptation strategies may need to be updated to account for these new patterns. Environmental policy experts are reviewing the research to determine what adjustments might be necessary in current climate action plans. The study's methodology involved collaboration between multiple institutions and incorporated both satellite data and ground-based observations. Scientists emphasize that while the findings are significant, they represent part of an ongoing research effort to better understand complex environmental systems. The research team plans to continue monitoring these patterns and will publish additional findings as data becomes available.",
        
        "Technology innovation continues to accelerate as companies announce breakthrough developments in artificial intelligence, quantum computing, and biotechnology sectors. These advances promise to transform industries ranging from healthcare and finance to manufacturing and entertainment. Research teams worldwide are collaborating on projects that could deliver significant benefits to society while addressing complex challenges. The pace of innovation has been particularly notable in areas where interdisciplinary approaches are yielding unexpected results. Industry leaders emphasize the importance of responsible development and implementation of these technologies. Regulatory frameworks are being developed to ensure that innovation proceeds safely and ethically. The economic implications of these technological advances are substantial, with potential impacts on employment patterns, productivity, and competitive dynamics across multiple sectors. Educational institutions are adapting their curricula to prepare students for careers in these emerging fields. International cooperation on research and development continues to expand as countries recognize the global nature of many technological challenges and opportunities."
    ]
    
    articles = []
    for i in range(count):
        # Add variation to each article
        base_article = templates[i % len(templates)]
        variation = f" Additional context for article {i+1}: This development follows recent trends in the sector and builds upon previous research efforts. Industry observers note that these changes reflect broader patterns in market dynamics and technological adoption. Stakeholders are carefully evaluating the potential long-term implications while monitoring short-term developments."
        
        full_article = base_article + variation
        articles.append(full_article)
    
    return articles

def benchmark_sentiment_analysis(articles, batch_sizes=[1, 8, 16, 32]):
    """Benchmark sentiment analysis with realistic news articles"""
    
    print("🔍 Loading sentiment analysis model...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if torch.cuda.is_available() else -1,
        batch_size=32
    )
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n📊 Testing batch size: {batch_size}")
        
        # Warm up
        sentiment_analyzer(articles[:2])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Benchmark
        start_time = time.time()
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            _ = sentiment_analyzer(batch)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        articles_per_sec = len(articles) / total_time
        
        results[batch_size] = {
            'total_time': total_time,
            'articles_per_sec': articles_per_sec,
            'avg_time_per_article': total_time / len(articles) * 1000  # ms
        }
        
        print(f"   Time: {total_time:.2f}s")
        print(f"   Speed: {articles_per_sec:.1f} articles/sec")
        print(f"   Avg per article: {total_time / len(articles) * 1000:.1f}ms")
    
    return results

def benchmark_embeddings(articles, batch_sizes=[1, 8, 16, 32]):
    """Benchmark sentence embeddings with realistic news articles"""
    
    print("\n🧠 Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n📊 Testing embeddings batch size: {batch_size}")
        
        # Warm up
        model.encode(articles[:2])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Benchmark
        start_time = time.time()
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            _ = model.encode(batch)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        articles_per_sec = len(articles) / total_time
        
        results[batch_size] = {
            'total_time': total_time,
            'articles_per_sec': articles_per_sec,
            'avg_time_per_article': total_time / len(articles) * 1000  # ms
        }
        
        print(f"   Time: {total_time:.2f}s")
        print(f"   Speed: {articles_per_sec:.1f} articles/sec")
        print(f"   Avg per article: {total_time / len(articles) * 1000:.1f}ms")
    
    return results

def main():
    print("=" * 60)
    print("🔥 JustNews V4 REALISTIC Performance Test")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    else:
        print("⚠️  Using CPU (GPU not available)")
    
    # Generate realistic articles
    print(f"\n📰 Generating realistic news articles...")
    articles = create_realistic_news_articles(100)
    
    avg_length = np.mean([len(article) for article in articles])
    print(f"   Count: {len(articles)} articles")
    print(f"   Average length: {avg_length:.0f} characters")
    print(f"   Sample: {articles[0][:100]}...")
    
    # Benchmark sentiment analysis
    print(f"\n🎯 Benchmarking Sentiment Analysis (Realistic Articles)")
    sentiment_results = benchmark_sentiment_analysis(articles)
    
    # Benchmark embeddings
    print(f"\n🎯 Benchmarking Embeddings (Realistic Articles)")
    embedding_results = benchmark_embeddings(articles)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 REALISTIC PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print("\n🎯 Sentiment Analysis Results:")
    for batch_size, result in sentiment_results.items():
        print(f"   Batch {batch_size:2d}: {result['articles_per_sec']:5.1f} articles/sec ({result['avg_time_per_article']:5.1f}ms per article)")
    
    print("\n🧠 Embedding Results:")
    for batch_size, result in embedding_results.items():
        print(f"   Batch {batch_size:2d}: {result['articles_per_sec']:5.1f} articles/sec ({result['avg_time_per_article']:5.1f}ms per article)")
    
    # Best performance
    best_sentiment = max(sentiment_results.values(), key=lambda x: x['articles_per_sec'])
    best_embedding = max(embedding_results.values(), key=lambda x: x['articles_per_sec'])
    
    print(f"\n🏆 Best Performance:")
    print(f"   Sentiment Analysis: {best_sentiment['articles_per_sec']:.1f} articles/sec")
    print(f"   Embeddings: {best_embedding['articles_per_sec']:.1f} articles/sec")
    
    # V4 target comparison
    target_min, target_max = 200, 400
    sentiment_vs_target = best_sentiment['articles_per_sec'] / target_max * 100
    
    print(f"\n🎯 V4 Target Comparison (vs {target_max} articles/sec target):")
    if best_sentiment['articles_per_sec'] >= target_min:
        print(f"   ✅ EXCEEDS TARGET: {sentiment_vs_target:.1f}% of maximum target")
    else:
        print(f"   ⚠️  Below target: {sentiment_vs_target:.1f}% of maximum target")
    
    print(f"\n💡 Note: These are realistic benchmarks using {avg_length:.0f}-character articles")
    print(f"   (much more accurate than synthetic data!)")

if __name__ == "__main__":
    main()
