#!/bin/bash
# Quick Performance Test for GPU Acceleration

echo "Running JustNews V4 GPU Performance Test"

cd /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment
source /home/nvidia/.venvs/rapids25.06_python3.12/bin/activate

# Test GPU acceleration directly
python -c "
import time
from hybrid_tools_v4 import GPUAcceleratedAnalyst

print('Testing GPU acceleration...')
gpu_analyzer = GPUAcceleratedAnalyst()

test_articles = [
    'Breaking: Major tech company announces revolutionary AI breakthrough.',
    'Political tensions rise as world leaders meet for emergency summit.',
    'Economic markets surge following positive employment data reports.'
]

start_time = time.time()
for i, article in enumerate(test_articles):
    sentiment = gpu_analyzer.score_sentiment_gpu(article)
    bias = gpu_analyzer.score_bias_gpu(article)
    print(f'Article {i+1}: Sentiment={sentiment:.3f}, Bias={bias:.3f}')

end_time = time.time()
total_time = end_time - start_time
articles_per_sec = len(test_articles) / total_time

print(f'')
print(f'Performance Results:')
print(f'Total time: {total_time:.3f}s')
print(f'Articles per second: {articles_per_sec:.1f}')
print(f'Target (42.1/sec): {'MET' if articles_per_sec > 30 else 'BELOW TARGET'}')
"
