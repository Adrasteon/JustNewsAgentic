"""Conservative, lint-clean GPU fact-checker helpers used for triage.

This module provides a small, well-indented implementation that avoids heavy
runtime dependencies when possible and is safe to lint/format during the
agents triage pass.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

MODEL_NAME = "microsoft/DialoGPT-large"
FEEDBACK_LOG = os.environ.get("FACT_CHECKER_FEEDBACK_LOG", "./feedback_fact_checker.log")

logger = logging.getLogger(__name__)


class GPUAcceleratedFactChecker:
    """Minimal GPU-accelerated fact checker used during triage pass."""

    def __init__(self) -> None:
        self.gpu_available: bool = False
        self.models_loaded: bool = False
        self.performance_stats: Dict[str, Any] = {
            "total_requests": 0,
            "gpu_requests": 0,
            "fallback_requests": 0,
            "total_time": 0.0,
            "average_time": 0.0,
        }
        self._initialize_models()

    def _initialize_models(self) -> None:
        if not HAS_TRANSFORMERS:
            return
        try:
            if torch.cuda.is_available():
                try:
                    # Attempt to create pipelines for GPU; if anything fails,
                    # fall back to CPU initialization.
                    self.fact_verification_pipeline = pipeline(
                        "text-generation", model=MODEL_NAME, device=0  # type: ignore[arg-type]
                    )
                    self.news_validation_pipeline = pipeline(
                        "text-classification", model="facebook/bart-large-mnli", device=0  # type: ignore[arg-type]
                    )
                    self.models_loaded = True
                    self.gpu_available = True
                except Exception:
                    logger.exception("Failed to initialize GPU pipelines; falling back to CPU")
                    self._initialize_cpu()
            else:
                self._initialize_cpu()
        except Exception:
            logger.exception("Model initialization unexpected failure")

    def _initialize_cpu(self) -> None:
        if not HAS_TRANSFORMERS:
            return
        try:
            self.cpu_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)  # type: ignore[attr-defined]
            self.cpu_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # type: ignore[attr-defined]
            self.cpu_pipeline = pipeline(
                "text-generation", model=self.cpu_model, tokenizer=self.cpu_tokenizer, device=-1  # type: ignore[arg-type]
            )
            self.models_loaded = True
        except Exception:
            logger.exception("CPU model initialization failed")

    def validate_is_news(self, content: str) -> Dict[str, Any]:
        start_time = datetime.now()
        try:
            if getattr(self, "gpu_available", False) and getattr(self, "models_loaded", False):
                res = self.news_validation_pipeline(content, ["news article", "opinion piece"])  # type: ignore[call-arg]
                labels = res.get("labels") if isinstance(res, dict) else []
                scores = res.get("scores") if isinstance(res, dict) else []
                is_news = bool(labels and labels[0] == "news article" and scores and scores[0] > 0.5)
                confidence = scores[0] if scores else 0.0
                method = "gpu_classification"
                self.performance_stats["gpu_requests"] += 1
            else:
                keywords = ["breaking", "report", "headline", "news"]
                count = sum(1 for k in keywords if k in content.lower())
                is_news = count > 0
                confidence = count / max(len(keywords), 1)
                method = "cpu_keywords"
                self.performance_stats["fallback_requests"] += 1

            elapsed = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(elapsed)
            return {
                "is_news": is_news,
                "confidence": confidence,
                "scores": {"keyword_match": confidence} if method == "cpu_keywords" else {},
                "method": method,
                "processing_time": elapsed,
            }
        except Exception:
            logger.exception("validate_is_news failed")
            return {"is_news": False, "confidence": 0.0, "scores": {}, "method": "error", "processing_time": 0.0}

    def verify_claims_batch(self, claims: List[str], sources: List[str]) -> Dict[str, Any]:
        try:
            if getattr(self, "gpu_available", False) and getattr(self, "models_loaded", False):
                return self._gpu_verify_claims(claims, sources)
            return self._cpu_verify_claims(claims, sources)
        except Exception:
            logger.exception("verify_claims_batch failed")
            return {"results": {c: "error" for c in claims}, "method": "error", "processing_time": 0.0}

    def _gpu_verify_claims(self, claims: List[str], sources: List[str]) -> Dict[str, Any]:
        start_time = datetime.now()
        results: Dict[str, str] = {}
        try:
            joined_sources = "\n".join(sources[:3])
            prompts = [f"Sources: {joined_sources}\nClaim: {c}\nVerified or not verified?" for c in claims]
            outputs = self.fact_verification_pipeline(prompts, max_new_tokens=16, do_sample=False)  # type: ignore[call-arg]
            for claim, out in zip(claims, outputs):
                text = out[0]["generated_text"] if isinstance(out, list) else out.get("generated_text", "")
                text = text.lower()
                if "verified" in text and "not verified" not in text:
                    results[claim] = "verified"
                elif "not verified" in text:
                    results[claim] = "not verified"
                else:
                    results[claim] = "uncertain"
        except Exception:
            logger.exception("_gpu_verify_claims failed; falling back to CPU")
            return self._cpu_verify_claims(claims, sources)
        elapsed = (datetime.now() - start_time).total_seconds()
        self._update_performance_stats(elapsed)
        self.performance_stats["gpu_requests"] += 1
        return {"results": results, "method": "gpu_dialogpt", "processing_time": elapsed}

    def _cpu_verify_claims(self, claims: List[str], sources: List[str]) -> Dict[str, Any]:
        start_time = datetime.now()
        results: Dict[str, str] = {}
        if getattr(self, "cpu_pipeline", None) is not None:
            try:
                joined_sources = "\n".join(sources[:2])
                for claim in claims:
                    prompt = f"Sources: {joined_sources}\nClaim: {claim}\nVerified or not verified?"
                    out = self.cpu_pipeline(prompt, max_new_tokens=8, do_sample=False)  # type: ignore[call-arg]
                    text = out[0]["generated_text"].lower() if out else ""
                    results[claim] = "verified" if ("verified" in text and "not" not in text) else "not verified"
            except Exception:
                logger.exception("CPU pipeline failed; using heuristic fallback")
                for claim in claims:
                    claim_words = set(claim.lower().split())
                    source_text = " ".join(sources).lower()
                    overlap = len(claim_words.intersection(set(source_text.split())))
                    threshold = len(claim_words) * 0.3
                    results[claim] = "verified" if overlap >= threshold else "not verified"
        else:
            for claim in claims:
                claim_words = set(claim.lower().split())
                source_text = " ".join(sources).lower()
                overlap = len(claim_words.intersection(set(source_text.split())))
                threshold = len(claim_words) * 0.3
                results[claim] = "verified" if overlap >= threshold else "not verified"
        elapsed = (datetime.now() - start_time).total_seconds()
        self._update_performance_stats(elapsed)
        self.performance_stats["fallback_requests"] += 1
        return {"results": results, "method": "cpu_fallback", "processing_time": elapsed}

    def _update_performance_stats(self, elapsed_time: float) -> None:
        self.performance_stats["total_requests"] += 1
        self.performance_stats["total_time"] += elapsed_time
        self.performance_stats["average_time"] = self.performance_stats["total_time"] / max(self.performance_stats["total_requests"], 1)

    def get_performance_stats(self) -> Dict[str, Any]:
        return {**self.performance_stats, "gpu_available": self.gpu_available, "models_loaded": self.models_loaded}


# Module-level helpers
_gpu_fact_checker: Optional[GPUAcceleratedFactChecker] = None


def get_gpu_fact_checker() -> GPUAcceleratedFactChecker:
    global _gpu_fact_checker
    if _gpu_fact_checker is None:
        _gpu_fact_checker = GPUAcceleratedFactChecker()
    return _gpu_fact_checker


def log_feedback(event: str, details: dict) -> None:
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")


def validate_is_news(content: str) -> bool:
    return get_gpu_fact_checker().validate_is_news(content).get("is_news", False)


def verify_claims(claims: List[str], sources: List[str]) -> Dict[str, str]:
    return get_gpu_fact_checker().verify_claims_batch(claims, sources).get("results", {c: "error" for c in claims})


def validate_is_news_detailed(content: str) -> Dict[str, Any]:
    return get_gpu_fact_checker().validate_is_news(content)


def verify_claims_detailed(claims: List[str], sources: List[str]) -> Dict[str, Any]:
    return get_gpu_fact_checker().verify_claims_batch(claims, sources)


def get_fact_checker_performance() -> Dict[str, Any]:
    return get_gpu_fact_checker().get_performance_stats()
