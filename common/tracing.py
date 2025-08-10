"""Optional OpenTelemetry tracing hooks for FastAPI services.

- If OpenTelemetry packages are not installed or OTLP endpoint is not set,
  this module becomes a no-op to keep production stable.
- When enabled, it adds an HTTP middleware that records request spans and
  attaches useful attributes (path, method, status, idempotency key).

Env:
- OTEL_EXPORTER_OTLP_ENDPOINT: e.g. http://otel-collector:4317 (gRPC) or 4318 (HTTP)
- OTEL_SERVICE_NAME: override service name if provided
"""
from __future__ import annotations

import os
import logging
from typing import Optional

from fastapi import FastAPI, Request

logger = logging.getLogger(__name__)

_tracing_enabled = False
_tracer = None  # type: ignore


def init_tracing(service_name: str) -> bool:
    """Initialize OpenTelemetry tracing if possible and configured.

    Returns True if tracing is enabled, False otherwise.
    """
    global _tracing_enabled, _tracer
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    service_override = os.environ.get("OTEL_SERVICE_NAME")
    if not endpoint:
        logger.debug("OTEL endpoint not set; tracing disabled")
        _tracing_enabled = False
        return False
    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore

        # Prefer HTTP exporter if available and endpoint looks like http(s),
        # otherwise fall back to gRPC exporter. Both imports are optional.
        exporter = None

        # HTTP exporter path (optional)
        http_exporter = None
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
                    OTLPSpanExporter as HttpOTLPSpanExporter,
                )
                http_exporter = HttpOTLPSpanExporter(endpoint=endpoint)
                exporter = http_exporter
            except Exception:
                http_exporter = None

        # gRPC exporter path (optional)
        if exporter is None:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
                    OTLPSpanExporter as GrpcOTLPSpanExporter,
                )
                # For gRPC, endpoint may be like "http://host:4317" or "host:4317"
                # The exporter accepts either; pass through as provided.
                exporter = GrpcOTLPSpanExporter(endpoint=endpoint)
            except Exception:
                exporter = None

        if exporter is None:
            logger.warning("OpenTelemetry exporters not available; tracing disabled")
            _tracing_enabled = False
            _tracer = None
            return False

        resource = Resource.create({
            "service.name": service_override or service_name,
        })
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_override or service_name)
        _tracing_enabled = True
        logger.info("OpenTelemetry tracing initialized for %s", service_name)
        return True
    except Exception as e:  # library missing or misconfigured
        logger.warning("OpenTelemetry not enabled: %s", e)
        _tracing_enabled = False
        _tracer = None
        return False


def add_tracing_middleware(app: FastAPI, service_name: str) -> None:
    """Attach a middleware that records per-request spans if tracing enabled."""

    @app.middleware("http")
    async def _otel_middleware(request: Request, call_next):  # type: ignore[no-redef]
        if not _tracing_enabled or _tracer is None:
            return await call_next(request)
        # Prepare attributes
        method = request.method
        path = request.url.path
        idem = request.headers.get("X-Idempotency-Key", "")
        with _tracer.start_as_current_span("http.request") as span:  # type: ignore[attr-defined]
            try:
                span.set_attribute("service.name", service_name)
                span.set_attribute("http.method", method)
                span.set_attribute("http.target", path)
                if idem:
                    span.set_attribute("idempotency.key", idem)
                response = await call_next(request)
                span.set_attribute("http.status_code", getattr(response, "status_code", 0))
                return response
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("exception.type", e.__class__.__name__)
                span.set_attribute("exception.message", str(e))
                raise
