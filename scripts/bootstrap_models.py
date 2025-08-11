#!/usr/bin/env python3
"""Bootstrap script to prefetch model weights for offline, local-first runs.

- Reads a manifest of model repository IDs (Hugging Face) per agent/component
- Downloads snapshots into TRANSFORMERS_CACHE (or provided --cache-dir)
- Writes a report with resolved revisions to training_system/bootstrap_report.json

Usage examples:
  python scripts/bootstrap_models.py --manifest scripts/model_manifest.example.json
  python scripts/bootstrap_models.py --subset synthesizer analyst
  python scripts/bootstrap_models.py --include-vision  # include large vision models

Environment:
- TRANSFORMERS_CACHE: preferred cache directory for model weights
- HF_HOME: optional Hugging Face home (token cache etc.)
- HF_TOKEN: optional token for gated or rate-limited repos

After bootstrap, set TRANSFORMERS_OFFLINE=1 to enforce offline operation.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any

try:
    from huggingface_hub import snapshot_download, HfApi
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "huggingface_hub is required. Install with: pip install huggingface_hub\n"
        f"Import error: {e}"
    )

logger = logging.getLogger("bootstrap_models")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


DEFAULT_MANIFEST = {
    "analyst": [
        "roberta-base"  # example baseline; replace with your production choice
    ],
    "synthesizer": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "facebook/bart-large-cnn",
        "google/flan-t5-base",
    ],
    "fact_checker": [
        "distilbert-base-uncased",
        "roberta-base",
        "sentence-transformers/all-MiniLM-L6-v2",
    ],
    "critic": [
        "microsoft/DialoGPT-medium"
    ],
    "newsreader_vision_optional": [
        # Vision models are large; include with --include-vision
        "llava-hf/llava-1.5-7b-hf",
        "Salesforce/blip2-opt-2.7b",
    ],
    "scout_optional": [
        # Add your scout models here if licensing allows local caching
    ],
}


def load_manifest(path: Path | None) -> Dict[str, List[str]]:
    if path is None:
        logger.info("Using built-in DEFAULT_MANIFEST")
        return DEFAULT_MANIFEST
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit("Manifest must be a JSON object {component: [repo_ids...]}")
    return {k: list(v) for k, v in data.items()}


def filter_manifest(
    manifest: Dict[str, List[str]],
    subset: List[str] | None,
    include_vision: bool,
) -> Dict[str, List[str]]:
    selected: Dict[str, List[str]] = {}
    keys = set(manifest.keys())

    if subset:
        missing = [k for k in subset if k not in keys]
        if missing:
            logger.warning("Subset entries not in manifest: %s", ", ".join(missing))
        for k in subset:
            if k in manifest:
                selected[k] = manifest[k]
    else:
        selected = dict(manifest)

    if not include_vision and "newsreader_vision_optional" in selected:
        logger.info("Excluding vision models (use --include-vision to include)")
        selected.pop("newsreader_vision_optional", None)

    return selected


def bootstrap(
    manifest: Dict[str, List[str]],
    cache_dir: Path,
    force: bool = False,
    ignore_errors: bool = False,
) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    results: Dict[str, Any] = {"cache_dir": str(cache_dir), "downloads": []}

    for component, repos in manifest.items():
        for repo_id in repos:
            try:
                logger.info("Downloading %s (%s)", repo_id, component)
                # Resolve latest revision for traceability
                try:
                    info = api.model_info(repo_id)  # type: ignore[arg-type]
                    revision = info.sha
                except Exception:
                    revision = None  # dataset or space or legacy; let snapshot decide
                local_path = snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    cache_dir=str(cache_dir),
                    local_files_only=False,
                    force_download=force,
                    allow_patterns=None,
                    ignore_patterns=None,
                    resume_download=True,
                )
                results["downloads"].append(
                    {
                        "component": component,
                        "repo_id": repo_id,
                        "revision": revision,
                        "local_path": local_path,
                    }
                )
            except Exception as e:
                msg = f"Failed to download {repo_id}: {e}"
                if ignore_errors:
                    logger.warning(msg)
                    results.setdefault("errors", []).append(
                        {"component": component, "repo_id": repo_id, "error": str(e)}
                    )
                    continue
                raise SystemExit(msg)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch model weights for offline runs")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to JSON manifest (defaults to built-in list)",
    )
    parser.add_argument(
        "--subset",
        nargs="*",
        default=None,
        help="Limit to specific components (keys from manifest)",
    )
    parser.add_argument(
        "--include-vision",
        action="store_true",
        help="Include large vision models (LLaVA/BLIP2)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("TRANSFORMERS_CACHE", ".cache/transformers")),
        help="Destination cache directory (defaults to TRANSFORMERS_CACHE or local .cache)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Continue on individual download errors",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("training_system/bootstrap_report.json"),
        help="Path to write a JSON report of downloads",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    selected = filter_manifest(manifest, args.subset, args.include_vision)

    logger.info("Using cache dir: %s", args.cache_dir)
    results = bootstrap(
        manifest=selected,
        cache_dir=args.cache_dir,
        force=args.force,
        ignore_errors=args.ignore_errors,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote report: %s", args.report)

    print(
        "\nBootstrap complete. You may now set TRANSFORMERS_OFFLINE=1 to enforce offline operation."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
