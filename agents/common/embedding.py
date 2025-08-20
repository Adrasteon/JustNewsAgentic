"""Shared embedding model helper.

Provides a process-local cached SentenceTransformer instance to avoid repeated
model downloads / loads and reduce GPU memory churn. Callers should prefer
this helper when they need a SentenceTransformer instance.
"""
from typing import Optional, Tuple
import logging
import os
import threading
from pathlib import Path
import inspect
import warnings

logger = logging.getLogger(__name__)

_MODEL_CACHE = {}
_MODEL_CACHE_LOCKS = {}
_SUPPRESSION_LOGGED = False

def get_shared_embedding_model(model_name: str = "all-MiniLM-L6-v2", cache_folder: Optional[str] = None, device: Optional[object] = None):
    """Return a shared SentenceTransformer instance for this process.

    Args:
        model_name: HF model id or local path.
        cache_folder: Optional cache folder passed to SentenceTransformer.
        device: Optional device spec (torch.device, 'cpu', 'cuda', 'cuda:0' or int GPU id).

    Returns:
        SentenceTransformer instance (from sentence_transformers).

    Raises:
        ImportError if sentence_transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        logger.error("sentence-transformers not available: %s", e)
        raise ImportError("sentence-transformers package is required to load embedding models")

    # Normalize cache folder: prefer explicit value, else detect agent caller and use agent-local models dir
    if cache_folder is None:
        # If caller is inside an agents/<agent>/ path, use that agent's models directory by default
        try:
            stack = inspect.stack()
            caller_agent = None
            for fr in stack:
                fname = str(fr.filename)
                parts = fname.split(os.path.sep)
                if 'agents' in parts:
                    idx = parts.index('agents')
                    if idx + 1 < len(parts):
                        caller_agent = parts[idx + 1]
                        break
            if caller_agent:
                cache_folder = os.environ.get(f"{caller_agent.upper()}_MODEL_CACHE") or f"./agents/{caller_agent}/models"
            else:
                cache_folder = os.environ.get("MEMORY_V2_CACHE", "./models/memory_v2")
        except Exception:
            cache_folder = os.environ.get("MEMORY_V2_CACHE", "./models/memory_v2")
    # Use absolute path for cache folder to avoid cache_key mismatches
    try:
        cache_folder = str(Path(cache_folder).expanduser().resolve())
    except Exception:
        cache_folder = str(cache_folder)

    # Normalize device key for caching; if device is None, auto-detect
    device_key = None
    try:
        import torch
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if isinstance(device, torch.device):
            device_key = str(device)
        elif isinstance(device, int):
            device_key = f"cuda:{device}"
        else:
            device_key = str(device)
    except Exception:
        # torch not available or other error; use string representation
        device_key = str(device) if device is not None else "auto"

    cache_key: Tuple[str, str, str] = (model_name, cache_folder or "", device_key)

    # Ensure there's a lock per cache key to avoid concurrent duplicate loads
    lock = _MODEL_CACHE_LOCKS.get(cache_key)
    if lock is None:
        lock = threading.Lock()
        _MODEL_CACHE_LOCKS[cache_key] = lock

    with lock:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

    # Ensure local model files exist under the agent cache when cache_folder is provided.
    # This guarantees agents write to their own ./agents/<agent>/models dirs.
    logger.info("Loading shared embedding model: %s (cache=%s) device=%s", model_name, cache_folder, device_key)
    if cache_folder:
        # If ensure_agent_model_exists is available, use it to guarantee a local model dir
        try:
            model_dir = ensure_agent_model_exists(model_name, cache_folder)
            model = SentenceTransformer(str(model_dir))
        except Exception:
            # Fallback: let SentenceTransformer handle download into cache_folder
            model = SentenceTransformer(model_name, cache_folder=cache_folder)
    else:
        model = SentenceTransformer(model_name)

    # Try to move to requested device if possible
    try:
        import torch
        if device is not None:
            # Accept torch.device, string, or int
            if isinstance(device, torch.device):
                target = device
            elif isinstance(device, int):
                target = torch.device(f"cuda:{device}")
            else:
                # string like 'cuda', 'cuda:0', 'cpu'
                target = torch.device(str(device))

            # Some SentenceTransformer wrappers accept .to(device)
            try:
                model = model.to(target)
            except Exception:
                # Fallback: some versions don't implement .to — ignore
                logger.debug("Could not .to() SentenceTransformer; continuing with default device")
    except Exception:
        # torch might not be available — ignore device
        pass

    # Wrap the SentenceTransformer to suppress known FutureWarnings from upstream
    # This suppression is controlled by the EMBEDDING_SUPPRESS_WARNINGS env var
    # so we can disable it for testing or while upgrading dependencies.
    # TODO: Remove suppression entirely after upstream libraries are updated.
    class _SentenceTransformerWrapper:
        """Proxy that delegates to a SentenceTransformer but filters noisy FutureWarnings

        Specifically suppresses warnings mentioning `encoder_attention_mask` which
        are emitted from some torch/transformers internals. We only filter this
        specific FutureWarning during encode() to avoid hiding other useful warnings.
        """
        def __init__(self, inner):
            self._inner = inner

        def encode(self, *args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=r".*encoder_attention_mask.*",
                )
                return self._inner.encode(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    suppress = os.environ.get('EMBEDDING_SUPPRESS_WARNINGS', '1') != '0'
    global _SUPPRESSION_LOGGED
    if suppress:
        wrapped = _SentenceTransformerWrapper(model)
        # Log once that we're suppressing an upstream FutureWarning so operators can track this
        if not _SUPPRESSION_LOGGED:
            logger.info("Embedding helper: suppressing known FutureWarning 'encoder_attention_mask' (EMBEDDING_SUPPRESS_WARNINGS=%s)",
                        os.environ.get('EMBEDDING_SUPPRESS_WARNINGS'))
            _SUPPRESSION_LOGGED = True
    else:
        # Return raw model (no suppression) for testing
        wrapped = model
    _MODEL_CACHE[cache_key] = wrapped
    return wrapped

def ensure_agent_model_exists(model_name: str, agent_cache_dir: str) -> str:
    """Ensure that a local copy of the model exists in agent_cache_dir.

    If the directory does not exist, this will download the model once using
    SentenceTransformer and save it into a temporary location then atomically
    move it into place. Uses a filesystem lock (flock) to avoid concurrent
    downloads across processes.

    Returns the absolute path to the local model directory.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        logger.error("sentence-transformers not available for ensure_agent_model_exists: %s", e)
        raise

    agent_cache_dir = str(Path(agent_cache_dir).expanduser().resolve())
    model_dir = Path(agent_cache_dir) / model_name.replace('/', '_')

    # If model already exists on disk, return immediately
    if model_dir.exists() and any(model_dir.iterdir()):
        return str(model_dir)

    # Ensure parent directory exists
    Path(agent_cache_dir).mkdir(parents=True, exist_ok=True)

    lock_path = str(model_dir) + '.lock'
    # Use an OS-level file lock so multiple processes coordinate safely
    import fcntl

    tmp_dir = Path(f"{model_dir}.tmp")
    try:
        with open(lock_path, 'w') as lf:
            # Block until we acquire exclusive lock
            try:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            except Exception:
                # If flock is unsupported, fall back to in-process threading lock
                lock = _MODEL_CACHE_LOCKS.get(lock_path)
                if lock is None:
                    lock = threading.Lock()
                    _MODEL_CACHE_LOCKS[lock_path] = lock
                lock.acquire()
                try:
                    # inside fallback lock
                    if model_dir.exists() and any(model_dir.iterdir()):
                        return str(model_dir)
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("Downloading model %s into temporary dir %s", model_name, tmp_dir)
                    model = SentenceTransformer(model_name, cache_folder=str(tmp_dir))
                    try:
                        model.save(str(tmp_dir))
                    except Exception:
                        pass
                    if model_dir.exists():
                        import shutil
                        shutil.rmtree(model_dir)
                    tmp_dir.replace(model_dir)
                    logger.info("Model %s downloaded and saved to %s", model_name, model_dir)
                    return str(model_dir)
                finally:
                    lock.release()

            # At this point we have flock on lf
            # Re-check under lock
            if model_dir.exists() and any(model_dir.iterdir()):
                return str(model_dir)

            try:
                tmp_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading model %s into temporary dir %s", model_name, tmp_dir)
                model = SentenceTransformer(model_name, cache_folder=str(tmp_dir))
                try:
                    model.save(str(tmp_dir))
                except Exception:
                    pass

                # Remove existing target if present and replace
                if model_dir.exists():
                    import shutil
                    shutil.rmtree(model_dir)
                tmp_dir.replace(model_dir)
                logger.info("Model %s downloaded and saved to %s", model_name, model_dir)
                return str(model_dir)
            except Exception as e:
                logger.error("Failed to ensure agent model exists for %s: %s", model_name, e)
                # Clean up tmp on failure
                try:
                    if tmp_dir.exists():
                        import shutil
                        shutil.rmtree(tmp_dir)
                except Exception:
                    pass
                raise
    except Exception as outer_e:
        logger.error("ensure_agent_model_exists outer failure for %s: %s", model_name, outer_e)
        try:
            if tmp_dir.exists():
                import shutil
                shutil.rmtree(tmp_dir)
        except Exception:
            pass
        raise
