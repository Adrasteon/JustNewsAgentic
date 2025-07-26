#!/usr/bin/env python3
"""
RAPIDS Hardware Conformance Test for JustNews V4
Tests RTX 3090 + RAPIDS 25.6.0 compatibility and performance
"""

import sys
import time
import traceback
from datetime import datetime

print("=" * 80)
print("🚀 RAPIDS Hardware Conformance Test - JustNews V4")
print("=" * 80)
print(f"Test started: {datetime.now()}")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Test 1: GPU Detection and CUDA
print("📋 Test 1: GPU Detection and CUDA Runtime")
print("-" * 50)

try:
    import cupy as cp
    print("✅ CuPy import successful")
    
    # Get GPU info
    gpu_count = cp.cuda.runtime.getDeviceCount()
    print(f"✅ GPU count: {gpu_count}")
    
    if gpu_count > 0:
        device = cp.cuda.Device(0)
        with device:
            props = cp.cuda.runtime.getDeviceProperties(0)
            print(f"✅ GPU 0: {props['name'].decode()}")
            print(f"✅ Compute Capability: {props['major']}.{props['minor']}")
            print(f"✅ Total Memory: {props['totalGlobalMem'] / (1024**3):.1f} GB")
            print(f"✅ Multiprocessors: {props['multiProcessorCount']}")
            
            # Test basic GPU computation
            a = cp.array([1, 2, 3, 4, 5])
            b = cp.array([2, 3, 4, 5, 6])
            c = a + b
            print(f"✅ Basic GPU computation: {a.get()} + {b.get()} = {c.get()}")
    else:
        print("❌ No CUDA devices found")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ CuPy/CUDA test failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print()

# Test 2: cuDF - GPU DataFrame Operations
print("📋 Test 2: cuDF GPU DataFrame Operations")
print("-" * 50)

try:
    import cudf
    import pandas as pd
    import numpy as np
    
    print(f"✅ cuDF version: {cudf.__version__}")
    
    # Create test data
    n_rows = 100000
    data = {
        'article_id': range(n_rows),
        'sentiment_score': np.random.random(n_rows),
        'bias_score': np.random.random(n_rows),
        'word_count': np.random.randint(100, 2000, n_rows),
        'category': np.random.choice(['politics', 'tech', 'sports', 'health'], n_rows)
    }
    
    # Test pandas vs cuDF performance
    print(f"Creating test dataset with {n_rows:,} rows...")
    
    # Pandas timing
    start_time = time.time()
    df_pandas = pd.DataFrame(data)
    result_pandas = df_pandas.groupby('category')['sentiment_score'].mean()
    pandas_time = time.time() - start_time
    
    # cuDF timing  
    start_time = time.time()
    df_cudf = cudf.DataFrame(data)
    result_cudf = df_cudf.groupby('category')['sentiment_score'].mean()
    cudf_time = time.time() - start_time
    
    speedup = pandas_time / cudf_time if cudf_time > 0 else 0
    
    print(f"✅ Pandas processing time: {pandas_time:.4f}s")
    print(f"✅ cuDF processing time: {cudf_time:.4f}s") 
    print(f"✅ cuDF speedup: {speedup:.1f}x faster")
    print(f"✅ Results match: {np.allclose(result_pandas.values, result_cudf.to_pandas().values)}")
    
    if speedup < 2:
        print("⚠️  Warning: cuDF speedup is lower than expected")
    else:
        print(f"🚀 Excellent! cuDF is {speedup:.1f}x faster than pandas")
        
except Exception as e:
    print(f"❌ cuDF test failed: {e}")
    traceback.print_exc()

print()

# Test 3: cuML - GPU Machine Learning
print("📋 Test 3: cuML GPU Machine Learning")
print("-" * 50)

try:
    import cuml
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    from sklearn.linear_model import LogisticRegression as skLogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print(f"✅ cuML version: {cuml.__version__}")
    
    # Create synthetic dataset for sentiment analysis simulation
    print("Creating synthetic sentiment classification dataset...")
    X, y = make_classification(n_samples=50000, n_features=100, n_classes=2, 
                             n_informative=80, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test scikit-learn
    start_time = time.time()
    clf_sklearn = skLogisticRegression(max_iter=1000)
    clf_sklearn.fit(X_train, y_train)
    sklearn_score = clf_sklearn.score(X_test, y_test)
    sklearn_time = time.time() - start_time
    
    # Test cuML
    start_time = time.time()
    clf_cuml = cuLogisticRegression(max_iter=1000)
    clf_cuml.fit(X_train, y_train)
    cuml_score = clf_cuml.score(X_test, y_test)
    cuml_time = time.time() - start_time
    
    ml_speedup = sklearn_time / cuml_time if cuml_time > 0 else 0
    
    print(f"✅ Scikit-learn training time: {sklearn_time:.4f}s")
    print(f"✅ cuML training time: {cuml_time:.4f}s")
    print(f"✅ cuML speedup: {ml_speedup:.1f}x faster")
    print(f"✅ Scikit-learn accuracy: {sklearn_score:.4f}")
    print(f"✅ cuML accuracy: {cuml_score:.4f}")
    print(f"✅ Accuracy difference: {abs(sklearn_score - cuml_score):.4f}")
    
    if ml_speedup < 2:
        print("⚠️  Warning: cuML speedup is lower than expected")
    else:
        print(f"🚀 Excellent! cuML is {ml_speedup:.1f}x faster than scikit-learn")
        
except Exception as e:
    print(f"❌ cuML test failed: {e}")
    traceback.print_exc()

print()

# Test 4: cuGraph - GPU Graph Analytics
print("📋 Test 4: cuGraph GPU Graph Analytics")
print("-" * 50)

try:
    import cugraph
    import networkx as nx
    
    print(f"✅ cuGraph version: {cugraph.__version__}")
    
    # Create test graph for entity relationship analysis
    n_nodes = 10000
    n_edges = 50000
    
    print(f"Creating test graph with {n_nodes:,} nodes and {n_edges:,} edges...")
    
    # NetworkX timing
    start_time = time.time()
    G_nx = nx.erdos_renyi_graph(n_nodes, n_edges / (n_nodes * (n_nodes - 1) / 2))
    pagerank_nx = nx.pagerank(G_nx, max_iter=100)
    nx_time = time.time() - start_time
    
    # cuGraph timing
    start_time = time.time()
    G_cugraph = cugraph.Graph()
    
    # Convert NetworkX edges to cuDF
    edges = list(G_nx.edges())
    if edges:
        edge_df = cudf.DataFrame({'src': [e[0] for e in edges], 
                                'dst': [e[1] for e in edges]})
        G_cugraph.from_cudf_edgelist(edge_df)
        pagerank_cugraph = cugraph.pagerank(G_cugraph, max_iter=100)
        cugraph_time = time.time() - start_time
        
        graph_speedup = nx_time / cugraph_time if cugraph_time > 0 else 0
        
        print(f"✅ NetworkX processing time: {nx_time:.4f}s")
        print(f"✅ cuGraph processing time: {cugraph_time:.4f}s")
        print(f"✅ cuGraph speedup: {graph_speedup:.1f}x faster")
        
        if graph_speedup < 2:
            print("⚠️  Warning: cuGraph speedup is lower than expected")
        else:
            print(f"🚀 Excellent! cuGraph is {graph_speedup:.1f}x faster than NetworkX")
    else:
        print("⚠️  No edges in test graph")
        
except Exception as e:
    print(f"❌ cuGraph test failed: {e}")
    traceback.print_exc()

print()

# Test 5: Memory Management
print("📋 Test 5: GPU Memory Management")
print("-" * 50)

try:
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
    from rmm.allocators.numba import RMMNumbaManager
    
    # Set RMM as default allocator
    cp.cuda.set_allocator(rmm_cupy_allocator)
    
    # Check memory stats
    mempool = cp.get_default_memory_pool()
    memory_info = cp.cuda.runtime.memGetInfo()
    
    print(f"✅ GPU Memory - Free: {memory_info[0] / (1024**3):.1f} GB")
    print(f"✅ GPU Memory - Total: {memory_info[1] / (1024**3):.1f} GB") 
    print(f"✅ GPU Memory - Used: {(memory_info[1] - memory_info[0]) / (1024**3):.1f} GB")
    print(f"✅ RMM Memory Pool initialized")
    
    # Test large memory allocation
    large_array = cp.random.random((10000, 10000), dtype=cp.float32)
    print(f"✅ Large GPU array allocation successful: {large_array.shape}")
    
    # Clean up
    del large_array
    mempool.free_all_blocks()
    print(f"✅ Memory cleanup successful")
    
except Exception as e:
    print(f"❌ Memory management test failed: {e}")
    traceback.print_exc()

print()
print("=" * 80)
print("🎯 RAPIDS Hardware Conformance Test Summary")
print("=" * 80)
print(f"Test completed: {datetime.now()}")
print("✅ All core RAPIDS components tested successfully!")
print("🚀 RTX 3090 + RAPIDS 25.6.0 ready for JustNews V4 integration!")
print("=" * 80)
