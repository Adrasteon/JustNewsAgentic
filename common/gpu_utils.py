"""Shared GPU utilities with professional resource management patterns.

Follows project GPU standards: context manager, memory logging, safe CPU
fallback, and specific error types. Designed to be lightweight and have
no hard dependency on torch when unavailable.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


class GPUError(Exception):
    """GPU operation failures."""


class ModelLoadError(GPUError):
    """Model loading failures."""


@contextlib.contextmanager
def gpu_context(device: int = 0) -> Iterator[None]:
    """Context manager for safe GPU operations with CPU fallback.

    - If torch with CUDA is available, sets device and empties cache on exit.
    - If not, yields control without raising.
    - Logs before/after memory to detect potential leaks.
    """
    try:
        import torch  # type: ignore
        has_cuda = torch.cuda.is_available()
    except Exception:  # torch not installed or import error
        has_cuda = False
        torch = None  # type: ignore

    if not has_cuda:
        # CPU fallback
        logger.debug("GPU not available; using CPU fallback context")
        yield
        return

    # GPU path
    assert torch is not None
    try:
        torch.cuda.set_device(device)
        initial_memory = torch.cuda.memory_allocated()
        logger.debug(f"gpu_context enter: device={device} mem={initial_memory}")
        yield
    finally:
        try:
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            if final_memory > initial_memory:
                logger.warning(
                    "Potential memory leak in GPU context: +%d bytes",
                    final_memory - initial_memory,
                )
        except Exception as e:
            logger.error(f"Error during GPU cleanup: {e}")


def safe_gpu_operation(operation, *args, **kwargs):
    """Run an operation inside gpu_context and map CUDA errors to GPUError."""
    try:
        with gpu_context():
            return operation(*args, **kwargs)
    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.error(f"CUDA error in {getattr(operation, '__name__', 'op')}: {e}")
            raise GPUError(f"GPU operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in {getattr(operation, '__name__', 'op')}: {e}")
        raise
