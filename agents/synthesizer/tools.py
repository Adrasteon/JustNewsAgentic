# Optimized Synthesizer Configuration  
# Phase 1 Memory Optimization: Context reduction + Lightweight embeddings

import os
import logging
from typing import List
from datetime import datetime

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None
try:
    from bertopic import BERTopic
except ImportError:
    BERTopic = None
try:
    import hdbscan
except ImportError:
    hdbscan = None

# PHASE 1 OPTIMIZATIONS APPLIED
MODEL_NAME = "microsoft/DialoGPT-medium"
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/dialogpt-medium")
EMBEDDING_MODEL = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")  # Lightweight embeddings
OPTIMIZED_MAX_LENGTH = 1024  # Reduced from 2048 (clustering tasks don't need full context)
OPTIMIZED_BATCH_SIZE = 4     # Memory-efficient for embeddings processing

FEEDBACK_LOG = os.environ.get("SYNTHESIZER_FEEDBACK_LOG", "./feedback_synthesizer.log")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("synthesizer.tools")

def get_dialog_model():
    """Load optimized DialoGPT-medium model with memory-efficient configuration."""
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError("transformers library is not installed.")
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        logger.info(f"Downloading {MODEL_NAME} to {MODEL_PATH}...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_PATH)
    else:
        logger.info(f"Loading {MODEL_NAME} from local cache {MODEL_PATH}...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return model, tokenizer

def get_embedding_model():
    """Load lightweight embedding model optimized for clustering."""
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers library is not installed.")
    return SentenceTransformer(EMBEDDING_MODEL)

def log_feedback(event: str, details: dict):
    """Log feedback for continual learning and retraining."""
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")

def cluster_articles(article_texts: List[str], n_clusters: int = 2) -> List[List[int]]:
    """Cluster articles using optimized embedding configuration."""
    if not article_texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(article_texts)
    method = os.environ.get("SYNTHESIZER_CLUSTER_METHOD", "kmeans").lower()
    clusters = []
    try:
        if method == "bertopic":
            if BERTopic is None:
                raise ImportError("BERTopic is not installed.")
            topic_model = BERTopic(verbose=False)
            topics, _ = topic_model.fit_transform(article_texts)
            n_topics = max(topics) + 1 if topics else 0
            clusters = [[] for _ in range(n_topics)]
            for idx, topic in enumerate(topics):
                if topic >= 0:
                    clusters[topic].append(idx)
        elif method == "hdbscan":
            if hdbscan is None:
                raise ImportError("hdbscan is not installed.")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
            labels = clusterer.fit_predict(embeddings)
            n_clusters = max(labels) + 1 if labels.size > 0 else 0
            clusters = [[] for _ in range(n_clusters)]
            for idx, label in enumerate(labels):
                if label >= 0:
                    clusters[label].append(idx)
        else:
            if KMeans is None:
                raise ImportError("sklearn KMeans is not installed.")
            if len(article_texts) < n_clusters:
                n_clusters = len(article_texts)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            clusters = [[] for _ in range(n_clusters)]
            for idx, label in enumerate(labels):
                clusters[label].append(idx)
        log_feedback("cluster_articles", {"method": method, "n_clusters": len(clusters), "clusters": clusters})
        return clusters
    except Exception as e:
        logger.error(f"Error in cluster_articles: {e}")
        log_feedback("cluster_articles_error", {"error": str(e), "method": method})
        return []

def neutralize_text(text: str) -> str:
    """Use optimized model to neutralize text with reduced memory usage."""
    model, tokenizer = get_dialog_model()
    if pipeline is None:
        raise ImportError("transformers pipeline is not available.")
    
    # Use optimized pipeline configuration
    pipe = pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer,
        max_length=OPTIMIZED_MAX_LENGTH,
        batch_size=OPTIMIZED_BATCH_SIZE
    )
    
    prompt = f"Neutralize the following text for bias and strong language: {text}"
    result = pipe(prompt, max_new_tokens=256)[0]["generated_text"]
    log_feedback("neutralize_text", {"input": text, "output": result})
    return result

def aggregate_cluster(article_texts: List[str]) -> str:
    """Use optimized model to summarize article clusters efficiently."""
    model, tokenizer = get_dialog_model()
    if pipeline is None:
        raise ImportError("transformers pipeline is not available.")
    
    # Use optimized pipeline configuration
    pipe = pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer,
        max_length=OPTIMIZED_MAX_LENGTH,
        batch_size=OPTIMIZED_BATCH_SIZE
    )
    
    joined = "\n".join(article_texts)
    prompt = f"Summarize the following articles into a neutral, concise summary: {joined}"
    result = pipe(prompt, max_new_tokens=512)[0]["generated_text"]
    log_feedback("aggregate_cluster", {"input": article_texts, "output": result})
    return result
