"""Lightweight observability utilities for all agents.

This module provides a minimal in-memory metrics collector and a FastAPI
middleware to record request counts and durations in a Prometheus-compatible
text exposition format, without external dependencies.

Notes:
- Output is text/plain; version=0.0.4
- Labels are captured internally but rendered as flat metric names for
  simplicity and zero-dependency operation.
"""
from __future__ import annotations

import time
import threading
from typing import Dict, Tuple, Optional, List

from fastapi import Request, FastAPI


class MetricsCollector:
    """Minimal counters and histograms for Prometheus text exposition.

    Attributes:
        agent: Short agent name prefix for metric names (e.g., "memory").
        counters: Mapping of (name, label tuples) to integer counts.
        histograms: Mapping of (name, label tuples) to histogram data.
        default_buckets: Default histogram buckets in seconds.
    """

    def __init__(self, agent: str) -> None:
        self.agent: str = agent
        self._lock: threading.Lock = threading.Lock()
        self.counters: Dict[Tuple[str, ...], int] = {}
        self.histograms: Dict[Tuple[str, ...], Dict[str, float]] = {}
        self.default_buckets: List[float] = [
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5
        ]

    def inc(self, name: str, labels: Optional[Dict[str, str]] = None, value: int = 1) -> None:
        """Increment a counter by value.

        Args:
            name: Metric base name (without agent prefix).
            labels: Optional labels (unused in rendering, kept for future use).
            value: Increment amount (default 1).
        """
        key = (name,) + tuple(sorted((labels or {}).items()))
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + value

    def observe(
        self,
        name: str,
        value_s: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> None:
        """Record a value into a histogram.

        Args:
            name: Histogram base name.
            value_s: Observation value in seconds.
            labels: Optional labels (unused in rendering, kept for future use).
            buckets: Custom bucket boundaries; defaults to self.default_buckets.
        """
        buckets = buckets or self.default_buckets
        key = (name,) + tuple(sorted((labels or {}).items()))
        with self._lock:
            h = self.histograms.get(key)
            if h is None:
                h = {f"bucket_le_{b}": 0 for b in buckets}
                h.update({"count": 0, "sum": 0.0})
                self.histograms[key] = h
            for b in buckets:
                if value_s <= b:
                    h[f"bucket_le_{b}"] += 1
            h["count"] += 1
            h["sum"] += value_s

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to a specific value."""
        key = (f"gauge_{name}",)
        with self._lock:
            # Store gauges in counters dict with a special prefix for simplicity
            self.counters[key] = float(value)  # type: ignore[assignment]

    def render(self) -> str:
        """Render all metrics into Prometheus text format lines.

        Returns:
            A newline-terminated string of metric lines.
        """
        lines: List[str] = []
        # Counters
        for key, value in self.counters.items():
            name = key[0]
            lines.append(f"{self.agent}_{name} {value}")
        # Histograms
        for key, h in self.histograms.items():
            name = key[0]
            for k, v in h.items():
                lines.append(f"{self.agent}_{name}_{k} {v}")
        return "\n".join(lines) + "\n"


def request_timing_middleware(app: FastAPI, metrics: MetricsCollector) -> FastAPI:
    """Install a simple timing middleware on a FastAPI app.

    Records per-request counts and durations into the provided collector.
    """

    @app.middleware("http")
    async def _mw(request: Request, call_next):  # type: ignore[no-redef]
        start = time.perf_counter()
        try:
            response = await call_next(request)
            return response
        except Exception:
            metrics.inc("errors_total")
            raise
        finally:
            dur = time.perf_counter() - start
            metrics.inc("requests_total")
            metrics.observe("request_duration_seconds", dur)

    return app
