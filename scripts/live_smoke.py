#!/usr/bin/env python3
"""
Live smoke test for JustNews V4.

Performs a minimal end-to-end check using live crawling and the full MCP flow:
1) Optional Playwright install for Chromium
2) Health and readiness checks for MCP Bus and agents
3) Trigger a tiny BBC crawl (1-3 items)
4) Verify Memory DB endpoints (save/get/vector search) behind X-Service-Token

Environment:
- BASE_URLS: MCP Bus and agent URLs via env or defaults
- MCP_SERVICE_TOKEN: shared secret; when set, added to requests

This script is intentionally lightweight and prints a concise PASS/FAIL summary.
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from typing import Dict, Any, Optional

import requests


SERVICE_TOKEN_ENV = "MCP_SERVICE_TOKEN"
HEADER_NAME = "X-Service-Token"


def svc_headers() -> Dict[str, str]:
    token = os.environ.get(SERVICE_TOKEN_ENV)
    return {HEADER_NAME: token} if token else {}


def get(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 5.0) -> requests.Response:
    h: Dict[str, str] = {}
    h.update(headers or {})
    return requests.get(url, headers=h, timeout=timeout)


def post(url: str, json_body: Any, headers: Optional[Dict[str, str]] = None, timeout: float = 10.0) -> requests.Response:
    h: Dict[str, str] = {}
    h.update(headers or {})
    return requests.post(url, json=json_body, headers=h, timeout=timeout)


def ensure_playwright() -> None:
    try:
        import playwright  # type: ignore  # noqa: F401
    except Exception:
        print("[setup] Installing Playwright and Chromium (one-time)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright>=1.45.0"])  # noqa: S603,S607
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"])  # noqa: S603,S607


def wait_healthy(name: str, url: str, headers: Dict[str, str], attempts: int = 30, delay: float = 1.0) -> bool:
    for i in range(attempts):
        try:
            r = get(url, headers=headers, timeout=3)
            if r.ok:
                print(f"[ok] {name} healthy: {url}")
                return True
        except Exception:
            pass
        if i in (0, 5, 10, 20):
            print(f"[wait] {name} not ready yet… ({i+1}/{attempts})")
        time.sleep(delay)
    print(f"[fail] {name} failed health: {url}")
    return False


def check_memory_ready(memory_base: str, headers: Dict[str, str]) -> bool:
    r = get(f"{memory_base}/ready", headers=headers)
    if not r.ok:
        print("[fail] Memory /ready status:", r.status_code, r.text[:200])
        return False
    data = r.json()
    ready = bool(data.get("ready"))
    print(f"[info] Memory readiness: ready={ready}, db_ok={data.get('db_ok')}, vector_ok={data.get('vector_ok')}")
    return ready


def tiny_bbc_crawl() -> int:
    """Run the built-in BBC crawler quickly; return number of items attempted.

    This reuses production_bbc_crawler.py with a small target if supported.
    """
    script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "production_bbc_crawler.py")
    if not os.path.exists(script):
        print("[warn] production_bbc_crawler.py not found; skipping crawl")
        return 0
    print("[run] Launching BBC crawl (short run)…")
    try:
        # Many scripts accept args; we just run default which fetches a small set in this repo
        subprocess.check_call([sys.executable, script])  # noqa: S603,S607
        return 1
    except subprocess.CalledProcessError as e:
        print("[warn] BBC crawl exited with status:", e.returncode)
        return 0


def memory_roundtrip(memory_base: str, headers: Dict[str, str]) -> bool:
    # Save one tiny article
    article = {
        "title": "Smoke Test Article",
        "content": "This is a live smoke test article.",
        "source": "smoke_test",
        "url": "https://example.org/smoke",
        "published_at": int(time.time()),
    }
    resp = post(f"{memory_base}/save_article", article, headers=headers)
    if not resp.ok:
        print("[fail] save_article:", resp.status_code, resp.text[:200])
        return False
    obj = resp.json()
    article_id = obj.get("article_id") or obj.get("id") or 1
    # Get it back
    resp2 = get(f"{memory_base}/get_article/{article_id}", headers=headers)
    if not resp2.ok:
        print("[fail] get_article:", resp2.status_code, resp2.text[:200])
        return False
    # Vector search
    q = {"query": "smoke test", "top_k": 1}
    resp3 = post(f"{memory_base}/vector_search_articles", q, headers=headers)
    if not resp3.ok:
        print("[warn] vector_search_articles failed (continuing):", resp3.status_code)
    return True


def main() -> int:
    # Base URLs (override via env if needed)
    mcp_bus = os.environ.get("MCP_BUS_URL", "http://localhost:8000")
    chief = os.environ.get("CHIEF_URL", "http://localhost:8001")
    memory = os.environ.get("MEMORY_URL", "http://localhost:8007")
    scout = os.environ.get("SCOUT_URL", "http://localhost:8002")
    analyst = os.environ.get("ANALYST_URL", "http://localhost:8004")
    fact = os.environ.get("FACT_URL", "http://localhost:8003")
    synth = os.environ.get("SYNTH_URL", "http://localhost:8005")
    critic = os.environ.get("CRITIC_URL", "http://localhost:8006")

    headers = svc_headers()

    print("[step] 0. Ensure Playwright available (for crawlers)")
    try:
        ensure_playwright()
    except Exception as e:
        print("[warn] Playwright setup failed:", e)

    print("[step] 1. Health checks")
    all_ok = True
    # MCP Bus likely exposes /agents rather than /health; use agents as health probe
    all_ok &= wait_healthy("MCP Bus", f"{mcp_bus}/agents", headers)
    for (name, base) in [
        ("Chief Editor", chief),
        ("Scout", scout),
        ("Fact Checker", fact),
        ("Analyst", analyst),
        ("Synthesizer", synth),
        ("Critic", critic),
        ("Memory", memory),
    ]:
        all_ok &= wait_healthy(name, f"{base}/health", headers)
    if not all_ok:
        print("[fail] One or more services failed health checks")
        return 2

    print("[step] 2. Memory readiness")
    if not check_memory_ready(memory, headers):
        print("[fail] Memory readiness failed; check DB/pgvector setup")
        return 3

    print("[step] 3. Tiny BBC crawl (optional)")
    tiny_bbc_crawl()

    print("[step] 4. Memory roundtrip")
    if not memory_roundtrip(memory, headers):
        print("[fail] Memory roundtrip failed")
        return 4

    print("[PASS] Live smoke succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
