#!/bin/bash
# REAL GPU Performance Test with Actual Models and Realistic Articles

echo "REAL JustNews V4 GPU Performance Test"
echo "Using actual GPU models with realistic news articles"

cd /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment
source /home/nvidia/.venvs/rapids25.06_python3.12/bin/activate

# Test with REAL models and REAL article lengths
python -c "
import time
from hybrid_tools_v4 import GPUAcceleratedAnalyst

print('🔍 Testing with REAL GPU models and REALISTIC articles...')

# Create realistic full news articles (not tiny fragments)
realistic_articles = [
    '''Breaking News: Tech Giant Announces Revolutionary AI Breakthrough
    
    In a groundbreaking announcement today, a major technology company revealed its latest artificial intelligence breakthrough that could fundamentally transform the healthcare industry. The new AI system, developed over three years by a team of researchers, demonstrates unprecedented accuracy in medical diagnosis and treatment recommendations.
    
    The company's Chief Technology Officer stated during a press conference that the AI system has achieved a 95% accuracy rate in clinical trials, surpassing human specialists in several key areas. The technology utilizes advanced machine learning algorithms and has been trained on millions of medical cases from leading hospitals around the world.
    
    Industry experts are calling this development a potential game-changer for medical care, particularly in underserved regions where access to specialist doctors is limited. The AI system can analyze medical scans, patient histories, and symptoms to provide detailed diagnostic recommendations within minutes.
    
    However, some medical professionals have expressed concerns about the implications of AI replacing human judgment in critical healthcare decisions. The debate over AI's role in medicine continues to evolve as technology advances.''',
    
    '''Political Tensions Escalate as World Leaders Convene Emergency Climate Summit
    
    World leaders from over 50 nations gathered today for an emergency climate summit amid growing environmental concerns and international tensions. The unprecedented meeting was called following a series of extreme weather events that have devastated multiple regions across the globe in recent months.
    
    The summit's primary focus centers on establishing immediate action plans to address rising global temperatures and the increasing frequency of natural disasters. Scientific data presented at the opening session revealed alarming trends in ice cap melting, sea level rise, and ecosystem disruption.
    
    Diplomatic sources indicate that negotiations have been challenging, with developing nations demanding greater financial support from industrialized countries to implement green technologies. The economic implications of rapid environmental policy changes have created significant political pressure on all participating governments.
    
    Environmental activists outside the summit venue have maintained continuous protests, calling for more aggressive action on carbon emissions and fossil fuel regulations. Their demonstrations have drawn international media attention and added urgency to the proceedings.
    
    The summit is expected to conclude with a comprehensive agreement on climate action, though early reports suggest that achieving consensus among all participating nations remains a significant challenge.''',
    
    '''Economic Markets Experience Surge Following Positive Employment Data
    
    Financial markets around the world experienced significant gains today after the release of unexpectedly positive employment statistics from major economies. The data revealed job creation numbers that exceeded analyst predictions by substantial margins, indicating stronger economic recovery than previously anticipated.
    
    The unemployment rate dropped to its lowest level in over two years, while wage growth showed consistent improvement across multiple sectors. Manufacturing, technology, and service industries all contributed to the positive employment trends, suggesting broad-based economic strengthening.
    
    Stock markets responded enthusiastically to the news, with major indices posting gains of over 3% in early trading. Financial analysts noted that the strong employment data supports expectations of continued economic expansion and consumer spending growth.
    
    Corporate earnings reports released simultaneously showed quarterly results that surpassed expectations for most major companies. This combination of positive employment and earnings data has boosted investor confidence and led to increased market activity.
    
    However, some economists cautioned that inflationary pressures could emerge if the current economic growth trajectory continues without appropriate monetary policy adjustments. Central bank officials are closely monitoring the situation.'''
]

print(f'📊 Testing with {len(realistic_articles)} realistic articles...')
print(f'Average article length: {sum(len(a) for a in realistic_articles) // len(realistic_articles)} characters')

try:
    gpu_analyzer = GPUAcceleratedAnalyst()
    print('✅ GPU Analyzer initialized')
    
    # Test individual processing (current method)
    start_time = time.time()
    individual_results = []
    
    for i, article in enumerate(realistic_articles):
        print(f'Processing article {i+1}/{len(realistic_articles)}...')
        sentiment = gpu_analyzer.score_sentiment_gpu(article)
        bias = gpu_analyzer.score_bias_gpu(article)
        individual_results.append({'sentiment': sentiment, 'bias': bias})
        print(f'  Sentiment: {sentiment:.3f}, Bias: {bias:.3f}')
    
    end_time = time.time()
    total_time = end_time - start_time
    articles_per_sec = len(realistic_articles) / total_time
    
    print(f'')
    print(f'📊 REAL Performance Results:')
    print(f'Total articles: {len(realistic_articles)}')
    print(f'Total time: {total_time:.3f}s')
    print(f'Average per article: {total_time/len(realistic_articles):.3f}s')
    print(f'Articles per second: {articles_per_sec:.1f}')
    print(f'Original target (42.1/sec): {'✅ MET' if articles_per_sec > 42 else '⚠️ BELOW TARGET'}')
    print(f'Realistic expectation: {'✅ GOOD' if articles_per_sec > 5 else '⚠️ NEEDS OPTIMIZATION'}')
    
except Exception as e:
    print(f'❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
"
