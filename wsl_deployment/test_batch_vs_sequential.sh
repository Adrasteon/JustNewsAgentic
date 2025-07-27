#!/bin/bash
# Test: Sequential vs Batch Processing Performance

echo "GPU Batch Processing vs Sequential Processing Test"

cd /mnt/c/Users/marti/JustNewsAgentic/wsl_deployment
source /home/nvidia/.venvs/rapids25.06_python3.12/bin/activate

python -c "
import time
from hybrid_tools_v4 import GPUAcceleratedAnalyst

# Test articles
articles = [
    'Breaking: Tech company announces major AI breakthrough in healthcare.',
    'Political summit addresses climate change with new international policies.',
    'Economic data shows strong job growth across multiple industry sectors.',
    'Sports championship results in unexpected victory for underdog team.',
    'Science research reveals promising developments in renewable energy technology.'
]

print(f'🧪 Testing Sequential vs Batch Processing')
print(f'Articles to process: {len(articles)}')
print(f'Average length: {sum(len(a) for a in articles)//len(articles)} chars')
print()

gpu_analyzer = GPUAcceleratedAnalyst()

# TEST 1: Sequential Processing (current method)
print('🔄 Test 1: Sequential Processing (one-by-one)')
start_time = time.time()
sequential_results = []

for i, article in enumerate(articles):
    sentiment = gpu_analyzer.score_sentiment_gpu(article)
    sequential_results.append(sentiment)

sequential_time = time.time() - start_time
sequential_rate = len(articles) / sequential_time

print(f'Sequential Time: {sequential_time:.3f}s')
print(f'Sequential Rate: {sequential_rate:.1f} articles/sec')
print()

# TEST 2: Batch Processing (if available)
print('⚡ Test 2: Checking for Batch Processing Capability')

# Check if we can use the sentiment analyzer directly for batches
try:
    if hasattr(gpu_analyzer, 'sentiment_analyzer'):
        print('✅ Found sentiment analyzer - testing batch processing')
        
        start_time = time.time()
        # Use the HuggingFace pipeline's batch capability
        batch_results = gpu_analyzer.sentiment_analyzer(articles)
        batch_time = time.time() - start_time
        batch_rate = len(articles) / batch_time
        
        print(f'Batch Time: {batch_time:.3f}s')
        print(f'Batch Rate: {batch_rate:.1f} articles/sec')
        print(f'Speedup: {batch_rate/sequential_rate:.1f}x faster')
        print()
        
        # Show results comparison
        print('📊 Results Comparison:')
        for i, (seq_result, batch_result) in enumerate(zip(sequential_results, batch_results)):
            batch_score = batch_result[0]['score'] if batch_result[0]['label'] == 'POSITIVE' else 1 - batch_result[0]['score']
            print(f'Article {i+1}: Sequential={seq_result:.3f}, Batch={batch_score:.3f}')
            
    else:
        print('❌ Direct batch processing not available with current setup')
        print('Would need to implement proper batch processing in GPUAcceleratedAnalyst')
        
except Exception as e:
    print(f'❌ Batch test failed: {e}')
    print('This shows why we need proper batch implementation')

print()
print('💡 Batch Processing Explanation:')
print('- Sequential: GPU processes each article individually')
print('- Batch: GPU processes all articles simultaneously')
print('- GPUs excel at parallel operations on multiple inputs')
print('- Expected improvement: 3-5x faster with proper batching')
"
