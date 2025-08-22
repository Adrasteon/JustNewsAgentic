# Optimized Synthesizer Configuration  
# Phase 1 Memory Optimization: Context reduction + Lightweight embeddings

import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

# Import Synthesizer V2 Engine
try:
    from agents.synthesizer.synthesizer_v2_engine import SynthesizerV2Engine
    SYNTHESIZER_V2_AVAILABLE = True
except ImportError as e:
    SYNTHESIZER_V2_AVAILABLE = False
    logging.error(f"❌ Synthesizer V2 Engine not available: {e}")

# Import Synthesizer V3 Production Engine
try:
    from agents.synthesizer.synthesizer_v3_production_engine import SynthesizerV3ProductionEngine
    SYNTHESIZER_V3_AVAILABLE = True
except ImportError as e:
    SYNTHESIZER_V3_AVAILABLE = False
    logging.error(f"❌ Synthesizer V3 Production Engine not available: {e}")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
try:
    # Use shared embedding helper to avoid repeated loads across agents
    SentenceTransformer = None
except Exception:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
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
MODEL_NAME = "distilgpt2"
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/distilgpt2")
EMBEDDING_MODEL = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")  # Lightweight embeddings
OPTIMIZED_MAX_LENGTH = 1024  # Reduced from 2048 (clustering tasks don't need full context)
OPTIMIZED_BATCH_SIZE = 4     # Memory-efficient for embeddings processing

FEEDBACK_LOG = os.environ.get("SYNTHESIZER_FEEDBACK_LOG", "./feedback_synthesizer.log")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("synthesizer.tools")

# Online Training Integration
try:
    from training_system import (
        initialize_online_training, add_training_feedback, add_user_correction
    )
    ONLINE_TRAINING_AVAILABLE = True
    
    # Initialize online training for Synthesizer with 40-example threshold
    initialize_online_training(update_threshold=40)  # Update after 40 examples for synthesis tasks
    logger.info("🎓 Online Training enabled for Synthesizer V2")
    
except ImportError:
    ONLINE_TRAINING_AVAILABLE = False
    logger.warning("⚠️ Online Training not available for Synthesizer")

# Initialize Synthesizer Engines globally
synthesizer_v2_engine = None
synthesizer_v3_engine = None

# V2 Engine Initialization
if SYNTHESIZER_V2_AVAILABLE:
    try:
        synthesizer_v2_engine = SynthesizerV2Engine()
        logger.info("🚀 Synthesizer V2 Engine initialized")
        status = synthesizer_v2_engine.get_model_status()
        logger.info(f"📊 Model Status: {status['total_models']}/5 models loaded")
        for model, loaded in status.items():
            if model != 'total_models':
                logger.info(f"   {'✅' if loaded else '❌'} {model}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Synthesizer V2 Engine: {e}")
        synthesizer_v2_engine = None

# V3 Production Engine Initialization
if SYNTHESIZER_V3_AVAILABLE:
    try:
        synthesizer_v3_engine = SynthesizerV3ProductionEngine()
        logger.info("🚀 Synthesizer V3 Production Engine initialized")
        status = synthesizer_v3_engine.get_model_status()
        logger.info(f"📊 V3 Production Status: {status['total_models']} models loaded")
        for model, loaded in status.items():
            if model != 'total_models' and isinstance(loaded, bool):
                logger.info(f"   {'✅' if loaded else '❌'} {model}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Synthesizer V3 Production Engine: {e}")
        synthesizer_v3_engine = None

def get_synthesizer_v2_engine():
    """Get V2 engine instance"""
    return synthesizer_v2_engine

def get_synthesizer_v3_engine():
    """Get V3 production engine instance"""
    return synthesizer_v3_engine

def get_dialog_model():
    """Load optimized DialoGPT (deprecated)-medium model with memory-efficient configuration."""
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
    """Return a shared embedding model for the synthesizer.

    This uses the process-local shared instance when available to avoid
    repeated downloads and high memory usage.
    """
    try:
        # Prefer the shared helper when available
        from agents.common.embedding import get_shared_embedding_model
        agent_cache = os.environ.get('SYNTHESIZER_MODEL_CACHE') or str(Path('./agents/synthesizer/models').resolve())
        return get_shared_embedding_model(EMBEDDING_MODEL, cache_folder=agent_cache)
    except Exception:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers library is not installed.")

        # Fallback: attempt to ensure a local copy exists then construct from the local path
        agent_cache = os.environ.get('SYNTHESIZER_MODEL_CACHE') or str(Path('./agents/synthesizer/models').resolve())
        try:
            from agents.common.embedding import ensure_agent_model_exists
            _ = ensure_agent_model_exists(EMBEDDING_MODEL, agent_cache)
            from agents.common.embedding import get_shared_embedding_model
            return get_shared_embedding_model(EMBEDDING_MODEL, cache_folder=agent_cache)
        except Exception:
            # Final fallback: use shared helper if possible
            from agents.common.embedding import get_shared_embedding_model
            return get_shared_embedding_model(EMBEDDING_MODEL, cache_folder=agent_cache)

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

# ==================== SYNTHESIZER V2 TRAINING-INTEGRATED METHODS ====================

def synthesize_content_v2(article_texts: List[str], synthesis_type: str = "aggregate") -> Dict[str, Any]:
    """
    V2 Content synthesis with training integration using 5-model architecture
    
    Args:
        article_texts: List of article texts to synthesize
        synthesis_type: Type of synthesis ('aggregate', 'summarize', 'neutralize', 'refine')
    
    Returns:
        Dict with synthesized content and metadata
    """
    if not synthesizer_v2_engine:
        logger.warning("⚠️ Synthesizer V2 Engine not available, falling back to legacy")
        return {"content": aggregate_cluster(article_texts), "method": "legacy", "confidence": 0.5}
    
    try:
        start_time = datetime.utcnow()
        result = {"method": "synthesizer_v2", "synthesis_type": synthesis_type, "input_count": len(article_texts)}
        
        if synthesis_type == "aggregate":
            # Use V2 engine content aggregation
            aggregated = synthesizer_v2_engine.aggregate_cluster_content(article_texts)
            result["content"] = aggregated.get("best_result", "")
            result["all_results"] = aggregated
            result["confidence"] = 0.9
            
        elif synthesis_type == "summarize":
            # Use BART summarization for each text then aggregate
            summaries = []
            for text in article_texts:
                summary = synthesizer_v2_engine.summarize_content_bart(text)
                summaries.append(summary)
            result["content"] = " ".join(summaries)
            result["individual_summaries"] = summaries
            result["confidence"] = 0.8
            
        elif synthesis_type == "neutralize":
            # Use T5 neutralization
            neutralized_texts = []
            for text in article_texts:
                neutralized = synthesizer_v2_engine.neutralize_text_t5(text)
                neutralized_texts.append(neutralized)
            result["content"] = " ".join(neutralized_texts)
            result["individual_neutralized"] = neutralized_texts
            result["confidence"] = 0.85
            
        elif synthesis_type == "refine":
            # Use DialoGPT (deprecated) refinement
            refined_texts = []
            for text in article_texts:
                refined = synthesizer_v2_engine.refine_content_dialogpt(text)
                refined_texts.append(refined)
            result["content"] = " ".join(refined_texts)
            result["individual_refined"] = refined_texts
            result["confidence"] = 0.75
            
        else:
            # Default to aggregation
            aggregated = synthesizer_v2_engine.aggregate_cluster_content(article_texts)
            result["content"] = aggregated.get("best_result", "")
            result["confidence"] = 0.7
        
        # Add performance metrics
        end_time = datetime.utcnow()
        result["processing_time"] = (end_time - start_time).total_seconds()
        result["timestamp"] = end_time.isoformat()
        
        # Log feedback for training
        log_feedback("synthesize_content_v2", {
            "synthesis_type": synthesis_type,
            "input_count": len(article_texts),
            "output_length": len(result["content"]),
            "confidence": result["confidence"],
            "processing_time": result["processing_time"]
        })
        
        # Add training feedback if available
        if ONLINE_TRAINING_AVAILABLE:
            try:
                add_training_feedback(
                    agent_name="synthesizer",
                    task_type=f"synthesis_{synthesis_type}",
                    input_text=str(article_texts),
                    predicted_output=result["content"],
                    actual_output=result["content"],  # For unsupervised learning
                    confidence=result["confidence"]
                )
            except Exception as e:
                logger.warning(f"⚠️ Training feedback failed: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Synthesizer V2 synthesis failed: {e}")
        # Fallback to legacy method
        return {
            "content": aggregate_cluster(article_texts),
            "method": "fallback",
            "error": str(e),
            "confidence": 0.3
        }

def cluster_and_synthesize_v2(article_texts: List[str], n_clusters: int = 2) -> Dict[str, Any]:
    """
    V2 Advanced clustering and synthesis with training integration
    
    Args:
        article_texts: List of article texts to cluster and synthesize
        n_clusters: Number of clusters to create
    
    Returns:
        Dict with clustered and synthesized content
    """
    if not synthesizer_v2_engine:
        logger.warning("⚠️ Synthesizer V2 Engine not available")
        return {"error": "V2 engine not available"}
    
    try:
        start_time = datetime.utcnow()
        
        # Advanced clustering with V2 engine
        clustering_result = synthesizer_v2_engine.cluster_articles_advanced(article_texts)
        clusters = clustering_result.get("clusters", [])
        
        # Synthesize content for each cluster
        synthesized_clusters = []
        for i, cluster_indices in enumerate(clusters):
            if not cluster_indices:
                continue
                
            cluster_texts = [article_texts[idx] for idx in cluster_indices]
            cluster_synthesis = synthesize_content_v2(cluster_texts, synthesis_type="aggregate")
            
            synthesized_clusters.append({
                "cluster_id": i,
                "article_indices": cluster_indices,
                "article_count": len(cluster_indices),
                "synthesized_content": cluster_synthesis["content"],
                "confidence": cluster_synthesis["confidence"]
            })
        
        end_time = datetime.utcnow()
        result = {
            "method": "synthesizer_v2_clustering",
            "total_articles": len(article_texts),
            "clusters_created": len(synthesized_clusters),
            "synthesized_clusters": synthesized_clusters,
            "clustering_metadata": clustering_result,
            "processing_time": (end_time - start_time).total_seconds(),
            "timestamp": end_time.isoformat()
        }
        
        # Training feedback
        if ONLINE_TRAINING_AVAILABLE:
            try:
                from numpy import mean
                add_training_feedback(
                    agent_name="synthesizer",
                    task_type="cluster_synthesis",
                    input_text=str(article_texts),
                    predicted_output=str(result),
                    actual_output=str(result),  # For unsupervised learning
                    confidence=mean([c["confidence"] for c in synthesized_clusters]) if synthesized_clusters else 0.5
                )
            except Exception as e:
                logger.warning(f"⚠️ Training feedback failed: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Cluster and synthesis V2 failed: {e}")
        return {"error": str(e), "method": "failed"}

def add_synthesis_correction(original_input: str, expected_output: str, synthesis_type: str = "aggregate") -> Dict[str, Any]:
    """
    Add user correction for synthesis training
    
    Args:
        original_input: Original input text
        expected_output: Expected synthesis output  
        synthesis_type: Type of synthesis task
    
    Returns:
        Dict with correction status
    """
    if not ONLINE_TRAINING_AVAILABLE:
        return {"status": "error", "message": "Online training not available"}
    
    try:
        add_user_correction(
            agent_name="synthesizer",
            task_type=f"synthesis_{synthesis_type}",
            input_text=original_input,
            incorrect_output="",  # We don't have the incorrect output
            correct_output=expected_output,
            priority=2  # High priority for synthesis corrections
        )
        
        logger.info(f"✅ Synthesis correction added: {synthesis_type}")
        return {"status": "success", "message": "Correction added successfully"}
        
    except Exception as e:
        logger.error(f"❌ Failed to add synthesis correction: {e}")
        return {"status": "error", "message": str(e)}

# ==================== END SYNTHESIZER V2 METHODS ====================

# ==================== SYNTHESIZER V3 PRODUCTION METHODS ====================

def synthesize_content_v3(article_texts: List[str], context: str = "news analysis") -> Dict[str, Any]:
    """
    V3 Production content synthesis with training integration using 4-model architecture
    
    Features:
    - BERTopic clustering with proper UMAP parameters
    - BART summarization with minimum text validation  
    - FLAN-T5 neutralization and refinement (DialoGPT (deprecated) replacement)
    - SentenceTransformers embeddings
    - Training system integration with feedback collection
    - Comprehensive error handling and fallbacks
    
    Args:
        article_texts: List of article texts to synthesize
        context: Context for synthesis (default: "news analysis")
    
    Returns:
        Dict with synthesis results and metadata
    """
    if not SYNTHESIZER_V3_AVAILABLE:
        logger.warning("V3 engine not available, falling back to V2")
        return synthesize_content_v2(article_texts, context)
    
    engine = get_synthesizer_v3_engine()
    if engine is None:
        logger.warning("V3 engine not initialized, falling back to V2")
        return synthesize_content_v2(article_texts, context)
    
    try:
        logger.info(f"🧠 V3 Production synthesis starting: {len(article_texts)} articles")
        
        # Use V3 cluster and synthesize method
        result = engine.cluster_and_synthesize(article_texts)
        
        if not result.get('success', False):
            logger.warning("V3 cluster and synthesize failed, using content aggregation")
            # Fallback to direct content aggregation
            aggregated = engine.aggregate_cluster_content(article_texts)
            result = {
                "synthesis": aggregated.get("best_result", ""),
                "clusters": [[i for i in range(len(article_texts))]],
                "n_clusters": 1,
                "articles_processed": len(article_texts),
                "success": True,
                "fallback_used": True,
                "aggregation_details": aggregated
            }
        
        # Add V3-specific metadata
        result.update({
            "engine": "v3_production",
            "method": "cluster_and_synthesize_v3",
            "context": context,
            "timestamp": datetime.utcnow().isoformat(),
            "model_info": {
                "bertopic": "clustering",
                "bart": "summarization", 
                "flan_t5": "neutralization_refinement",
                "embeddings": "semantic_analysis"
            }
        })
        
        # Log feedback for training
        log_feedback("synthesize_content_v3", {
            "articles_processed": len(article_texts),
            "clusters_found": result.get('n_clusters', 1),
            "synthesis_length": len(result.get('synthesis', '')),
            "success": result.get('success', False),
            "context": context
        })
        
        # Add training feedback if available
        if ONLINE_TRAINING_AVAILABLE:
            try:
                add_training_feedback(
                    agent_name="synthesizer",
                    task_type=f"synthesis_v3_{context}",
                    input_text=str(article_texts),  # Convert to string format
                    predicted_output=result.get('synthesis', ''),
                    actual_output=result.get('synthesis', ''),  # No ground truth yet
                    confidence=result.get('confidence', 0.8)
                )
            except Exception as e:
                logger.warning(f"⚠️ V3 Training feedback failed: {e}")
        
        logger.info(f"✅ V3 Production synthesis completed: {len(result.get('synthesis', ''))} chars")
        return result
        
    except Exception as e:
        logger.error(f"❌ V3 Production synthesis failed: {e}")
        # Fallback to V2 if available
        if SYNTHESIZER_V2_AVAILABLE:
            logger.info("🔄 Falling back to V2 synthesis")
            return synthesize_content_v2(article_texts, context)
        else:
            return {"error": str(e), "method": "v3_failed_no_fallback", "success": False}


def cluster_and_synthesize_v3(article_texts: List[str], max_clusters: int = 5, context: str = "news analysis") -> Dict[str, Any]:
    """
    V3 Production advanced clustering and synthesis with training integration
    
    Features:
    - Advanced BERTopic clustering with proper UMAP configuration
    - Per-cluster BART summarization
    - FLAN-T5 cross-cluster synthesis and refinement
    - Training feedback collection
    - Robust error handling with V2 fallback
    
    Args:
        article_texts: List of article texts to cluster and synthesize
        max_clusters: Maximum number of clusters to create
        context: Context for synthesis
    
    Returns:
        Dict with clustering results and synthesized content
    """
    if not SYNTHESIZER_V3_AVAILABLE:
        logger.warning("V3 engine not available, falling back to V2")
        return cluster_and_synthesize_v2(article_texts, max_clusters, context)
    
    engine = get_synthesizer_v3_engine()
    if engine is None:
        logger.warning("V3 engine not initialized, falling back to V2")
        return cluster_and_synthesize_v2(article_texts, max_clusters, context)
    
    try:
        logger.info(f"🎯 V3 Production clustering starting: {len(article_texts)} articles, max {max_clusters} clusters")
        
        # Step 1: Advanced clustering
        cluster_results = engine.cluster_articles_advanced(article_texts)
        
        if not cluster_results.get('success', False):
            logger.warning("V3 clustering failed, using fallback approach")
            # Fallback: treat all as one cluster
            result = {
                "clusters": [[i for i in range(len(article_texts))]],
                "n_clusters": 1,
                "cluster_labels": ["general"] * len(article_texts),
                "synthesis": "",
                "articles_processed": len(article_texts),
                "success": False,
                "fallback_used": True
            }
        else:
            result = cluster_results.copy()
        
        # Step 2: Synthesize content for each cluster
        cluster_syntheses = []
        clusters = result.get('clusters', [[i for i in range(len(article_texts))]])
        
        for i, cluster_indices in enumerate(clusters):
            if not cluster_indices:
                continue
                
            cluster_articles = [article_texts[idx] for idx in cluster_indices if idx < len(article_texts)]
            if not cluster_articles:
                continue
                
            logger.info(f"🔄 Processing cluster {i+1}: {len(cluster_articles)} articles")
            
            # Use V3 content aggregation for this cluster
            cluster_synthesis = engine.aggregate_cluster_content(cluster_articles)
            cluster_syntheses.append({
                "cluster_id": i,
                "cluster_size": len(cluster_articles),
                "synthesis": cluster_synthesis.get("best_result", ""),
                "synthesis_details": cluster_synthesis
            })
        
        # Step 3: Final cross-cluster synthesis using FLAN-T5
        if cluster_syntheses:
            all_cluster_syntheses = [cs["synthesis"] for cs in cluster_syntheses if cs["synthesis"]]
            if all_cluster_syntheses:
                try:
                    # Use FLAN-T5 for final refinement
                    final_synthesis = engine.refine_content_flan_t5(
                        " ".join(all_cluster_syntheses), 
                        context=f"multi-cluster {context}"
                    )
                    result["synthesis"] = final_synthesis
                except Exception as e:
                    logger.warning(f"FLAN-T5 final synthesis failed: {e}")
                    result["synthesis"] = " ".join(all_cluster_syntheses)
            else:
                result["synthesis"] = ""
        else:
            result["synthesis"] = ""
        
        # Add V3-specific metadata
        result.update({
            "engine": "v3_production",
            "method": "cluster_and_synthesize_v3",
            "context": context,
            "max_clusters": max_clusters,
            "cluster_syntheses": cluster_syntheses,
            "timestamp": datetime.utcnow().isoformat(),
            "success": len(result.get("synthesis", "")) > 0
        })
        
        # Training feedback
        log_feedback("cluster_and_synthesize_v3", {
            "articles_processed": len(article_texts),
            "clusters_found": result.get('n_clusters', 1),
            "synthesis_length": len(result.get('synthesis', '')),
            "max_clusters": max_clusters,
            "success": result.get('success', False),
            "context": context
        })
        
        if ONLINE_TRAINING_AVAILABLE:
            try:
                add_training_feedback(
                    agent_name="synthesizer",
                    task_type=f"clustering_v3_{context}",
                    input_text=str(article_texts),  # Convert to string format
                    predicted_output=result.get('synthesis', ''),
                    actual_output=result.get('synthesis', ''),  # No ground truth yet
                    confidence=result.get('confidence', 0.8)
                )
            except Exception as e:
                logger.warning(f"⚠️ V3 Clustering training feedback failed: {e}")
        
        logger.info(f"✅ V3 Production clustering completed: {result.get('n_clusters', 1)} clusters, {len(result.get('synthesis', ''))} chars")
        return result
        
    except Exception as e:
        logger.error(f"❌ V3 Production clustering failed: {e}")
        # Fallback to V2 if available
        if SYNTHESIZER_V2_AVAILABLE:
            logger.info("🔄 Falling back to V2 clustering")
            return cluster_and_synthesize_v2(article_texts, max_clusters, context)
        else:
            return {"error": str(e), "method": "v3_clustering_failed", "success": False}


def add_synthesis_correction_v3(original_input: str, expected_output: str, synthesis_type: str = "aggregate") -> Dict[str, Any]:
    """
    Add user correction for V3 synthesis training
    
    Args:
        original_input: Original input text
        expected_output: Expected synthesis output  
        synthesis_type: Type of synthesis task
    
    Returns:
        Dict with correction status
    """
    if not ONLINE_TRAINING_AVAILABLE:
        return {"status": "error", "message": "Online training not available"}
    
    try:
        add_user_correction(
            agent_name="synthesizer",
            task_type=f"synthesis_v3_{synthesis_type}",
            input_text=original_input,
            incorrect_output="",  # We don't have the incorrect output
            correct_output=expected_output,
            priority=2  # High priority for synthesis corrections
        )
        
        logger.info(f"✅ V3 Synthesis correction added: {synthesis_type}")
        return {"status": "success", "message": "V3 correction added successfully", "engine": "v3_production"}
        
    except Exception as e:
        logger.error(f"❌ Failed to add V3 synthesis correction: {e}")
        return {"status": "error", "message": str(e), "engine": "v3_production"}


def get_synthesizer_status() -> Dict[str, Any]:
    """
    Get comprehensive status of both V2 and V3 synthesizer engines
    
    Returns:
        Dict with status of all available engines
    """
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "v2_available": SYNTHESIZER_V2_AVAILABLE,
        "v3_available": SYNTHESIZER_V3_AVAILABLE,
        "training_available": ONLINE_TRAINING_AVAILABLE,
        "engines": {}
    }
    
    # V2 Status
    if SYNTHESIZER_V2_AVAILABLE and synthesizer_v2_engine:
        try:
            v2_status = synthesizer_v2_engine.get_model_status()
            status["engines"]["v2"] = {
                "initialized": True,
                "models": v2_status,
                "architecture": "5_model_legacy"
            }
        except Exception as e:
            status["engines"]["v2"] = {"initialized": False, "error": str(e)}
    elif SYNTHESIZER_V2_AVAILABLE:
        status["engines"]["v2"] = {"initialized": False, "available": True}
    
    # V3 Status  
    if SYNTHESIZER_V3_AVAILABLE and synthesizer_v3_engine:
        try:
            v3_status = synthesizer_v3_engine.get_model_status()
            status["engines"]["v3"] = {
                "initialized": True,
                "models": v3_status,
                "architecture": "4_model_production"
            }
        except Exception as e:
            status["engines"]["v3"] = {"initialized": False, "error": str(e)}
    elif SYNTHESIZER_V3_AVAILABLE:
        status["engines"]["v3"] = {"initialized": False, "available": True}
    
    # Recommend primary engine
    if SYNTHESIZER_V3_AVAILABLE and synthesizer_v3_engine:
        status["recommended_engine"] = "v3_production"
    elif SYNTHESIZER_V2_AVAILABLE and synthesizer_v2_engine:
        status["recommended_engine"] = "v2_legacy"
    else:
        status["recommended_engine"] = "none_available"
    
    return status

# ==================== END SYNTHESIZER V3 PRODUCTION METHODS ====================