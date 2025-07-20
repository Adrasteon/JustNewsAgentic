
import os
import logging
from typing import List
from datetime import datetime

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.pipelines import pipeline
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.cluster import KMeans
    try:
        from bertopic import BERTopic
    except ImportError:
        BERTopic = None
    try:
        import hdbscan
    except ImportError:
        hdbscan = None
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
    SentenceTransformer = None
    np = None
    KMeans = None
    BERTopic = None
    hdbscan = None

MODEL_NAME = "meta-llama/Llama-3-70B-Instruct"
MODEL_PATH = os.environ.get("LLAMA_3_70B_PATH", "./models/llama-3-70b-instruct")
EMBEDDING_MODEL = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
FEEDBACK_LOG = os.environ.get("SYNTHESIZER_FEEDBACK_LOG", "./feedback_synthesizer.log")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("synthesizer.tools")

def get_llama_model():
    """
    Load the Llama-3-70B-Instruct model and tokenizer for text generation tasks.
    Downloads from Hugging Face if not cached locally.
    Returns:
        model, tokenizer: Hugging Face model and tokenizer objects.
    Raises:
        ImportError: If transformers is not installed.
    """
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
    """
    Load the sentence-transformers embedding model for clustering and semantic tasks.
    Returns:
        SentenceTransformer: Embedding model instance.
    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers library is not installed.")
    return SentenceTransformer(EMBEDDING_MODEL)

def log_feedback(event: str, details: dict):
    """
    Log feedback for continual learning and retraining.
    Feedback is logged in a standardized format (UTC timestamp, event, details) to a file for use in online or scheduled retraining.
    Args:
        event (str): The event or tool name.
        details (dict): Details about the event, input, output, or error.
    """
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")

def cluster_articles(article_texts: List[str], n_clusters: int = 2) -> List[List[int]]:
    """
    Cluster articles using sentence embeddings and a configurable clustering algorithm.
    Supported methods: KMeans (default), BERTopic, HDBSCAN (set via SYNTHESIZER_CLUSTER_METHOD env var).
    Args:
        article_texts (List[str]): List of article texts to cluster.
        n_clusters (int): Number of clusters (for KMeans; ignored for BERTopic/HDBSCAN).
    Returns:
        List[List[int]]: List of clusters, each a list of indices into article_texts.
    Feedback:
        Logs clustering method, number of clusters, and cluster assignments for retraining.
    Raises:
        ImportError: If required clustering library is not installed.
    """
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
    """
    Use the LLM to neutralize text (style transfer to neutral tone).
    Args:
        text (str): Input text to neutralize.
    Returns:
        str: Neutralized text output.
    Feedback:
        Logs input and output for continual learning and retraining.
    Raises:
        ImportError: If transformers pipeline is not available.
    """
    model, tokenizer = get_llama_model()
    if pipeline is None:
        raise ImportError("transformers pipeline is not available.")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    prompt = f"Neutralize the following text for bias and strong language: {text}"
    result = pipe(prompt, max_new_tokens=256)[0]["generated_text"]
    log_feedback("neutralize_text", {"input": text, "output": result})
    return result

def aggregate_cluster(article_texts: List[str]) -> str:
    """
    Use the LLM to summarize a cluster of articles into a neutral, concise summary.
    Args:
        article_texts (List[str]): List of article texts to summarize.
    Returns:
        str: Aggregated summary text.
    Feedback:
        Logs input articles and output summary for retraining and continual learning.
    Raises:
        ImportError: If transformers pipeline is not available.
    """
    model, tokenizer = get_llama_model()
    if pipeline is None:
        raise ImportError("transformers pipeline is not available.")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    joined = "\n".join(article_texts)
    prompt = f"Summarize the following articles into a neutral, concise summary: {joined}"
    result = pipe(prompt, max_new_tokens=512)[0]["generated_text"]
    log_feedback("aggregate_cluster", {"input": article_texts, "output": result})
    return result
