---
title: Needed-for-live-run — End-to-end live test checklist
status: Runbook (Aug 2025)
---

## Scope and success criteria

- [ ] All services start and pass health checks
- [ ] Memory agent /ready returns ready=true and vector_ok=true
- [ ] At least one live-scraped article completes the full workflow and is stored
- [ ] MCP Bus metrics render; circuit-breaker gauges present per agent
- [ ] No auth errors between services; inter-service requests carry X-Service-Token
- [ ] GPU acceleration active for agents that support it (CPU used only if GPU is unavailable)

## 1) Pre-flight environment

- [ ] OS: Ubuntu 24.04 with systemd (no Docker)
- [ ] Python: 3.11 or 3.12 available in the active environment
- [ ] Network egress to target news sites and model hubs
- [ ] GPU preferred: ensure NVIDIA drivers and CUDA 12.1+ are installed and working (nvidia-smi ok); CPU is used only if GPU use is not possible
- [ ] Local models only: plan for offline operation after initial bootstrap (no external inference)

## 2) Security and secrets

- [ ] Set a shared inter-service token in all services:
  - MCP_SERVICE_TOKEN=your-strong-secret
- [ ] Optional tokens (set where applicable):
  - HF_TOKEN (Hugging Face) for faster model pulls/caching
  - SERPAPI_KEY if any crawler path depends on it

## 2a) Bootstrap model weights (local-first)

- [ ] Prefetch all required model weights into your local cache before switching to offline mode.
  - Script: scripts/bootstrap_models.py
  - Example manifest: scripts/model_manifest.example.json
  - Default cache: TRANSFORMERS_CACHE or .cache/transformers
- [ ] Steps:
  ```bash
  # Ensure deps (if not already in your env)
  python -m pip install --upgrade huggingface_hub

  # Optional: set token for gated/private models or higher rate limits
  export HF_TOKEN=your_hf_token

  # Run bootstrap with the example manifest
  python scripts/bootstrap_models.py --manifest scripts/model_manifest.example.json

  # Optional: include vision models (large)
  # python scripts/bootstrap_models.py --include-vision --manifest scripts/model_manifest.example.json
  ```
- [ ] Verify report written to training_system/bootstrap_report.json and then set TRANSFORMERS_OFFLINE=1 for subsequent runs.

Note on versioning (dev vs prod)
- During development, floating to latest is acceptable; ensure the resolved revision SHA is captured in the bootstrap report and logs.
- For production, pin per agent to exact SHAs (lock manifest checked into repo). Runtime should load only the pinned SHA; avoid ambiguous tags like "main".

## 3) Database and vector store

- [ ] PostgreSQL 16 is reachable; env provided to Memory:
  - POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
- [ ] Enable pgvector on the target database:
  - CREATE EXTENSION IF NOT EXISTS vector;
- [ ] Initialize schema (one of):
  - Use Memory /db/init with a schema path, or
  - Apply SQL migrations directly if you manage DDL externally
- [ ] Verify readiness:
  - Memory /db/health returns db_ok=true
  - Memory /ready returns ready=true and vector_ok=true

## 4) Playwright and crawler prerequisites

- [ ] Install Playwright and Chromium (headless-friendly):
  ```bash
  python -m pip install "playwright>=1.45.0"
  python -m playwright install --with-deps chromium
  ```
- [ ] Confirm outbound access to BBC and your target sources

## 5) Start services (systemd)

- [ ] Launch all services (native, no Docker):
  ```bash
  deploy/systemd/enable_all.sh --fresh
  ```
- [ ] Verify ports are free and bound:
  - MCP Bus 8000; Chief 8001; Scout 8002; Fact 8003; Analyst 8004; Synth 8005; Critic 8006; Memory 8007; Reasoning 8008

## 6) Health, readiness, warmup

- [ ] MCP Bus health proxy: GET /agents (lists registered agents)
- [ ] Agents health: GET /health on each agent
- [ ] Memory readiness: GET /ready (db_ok/vector_ok)
- [ ] Optional: POST /warmup on agents to pre-populate light caches

## 7) Inter-service auth verification

- [ ] Ensure every outbound service call includes X-Service-Token
- [ ] Watch logs for 401/403; none should appear during internal calls

## 8) Live smoke (fast path)

- [ ] Run the provided live smoke script from repo root:
  ```bash
  # Optional: override URLs if services run on different hosts
  # MCP_BUS_URL=... CHIEF_URL=... SCOUT_URL=... FACT_URL=... ANALYST_URL=... SYNTH_URL=... CRITIC_URL=... MEMORY_URL=...
  python scripts/live_smoke.py
  ```
- One-shot execution (bootstrap → playwright → start services → smoke):
  ```bash
  bash scripts/execute_live_run.sh
  # or, skip bootstrap and reuse running services
  bash scripts/execute_live_run.sh --skip-bootstrap --no-restart
  ```
- What it does:
  - Ensures Playwright is installed
  - Probes MCP Bus (/agents) and agents (/health)
  - Verifies Memory readiness (/ready)
  - Optionally runs a short BBC crawl
  - Performs a Memory roundtrip: /save_article → /get_article/{id} → /vector_search_articles
- PASS criteria: script ends with "[PASS] Live smoke succeeded"

## 9) Observability and limits

- [ ] Metrics: scrape /metrics from MCP Bus and Memory (and others if exposed)
  - Per-agent gauges exported with name suffixes: circuit_breaker_open_<agent>, etc.
  - Idempotency cache size should be reported
- [ ] Optional tracing:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http(s)://collector:4318 or host:4317
  - OTEL_SERVICE_NAME override per service if desired
- [ ] Rate limits: MCP Bus caps on /register and /call (adjust before large runs)

## 10) Troubleshooting quick tips

- Health fails: check systemd service logs; confirm MCP_SERVICE_TOKEN matches across services
- Memory not ready: verify DB connectivity and pgvector loaded; re-check /db/health and /ready
- Crawl issues: re-run Playwright install with --with-deps; validate network/proxy
- Model pulls slow: set HF_TOKEN and TRANSFORMERS_CACHE/HF_HOME to a writable path
 - Enforce offline after bootstrap: set `TRANSFORMERS_OFFLINE=1` and ensure caches/checkpoints exist
- 429s from MCP Bus: lower crawl rate, increase limits conservatively

## 11) Safe rollback

- [ ] Stop services cleanly when done:
  ```bash
  ./stop_services.sh
  ```

## Appendix — Environment variables summary

- Inter-service auth: MCP_SERVICE_TOKEN
- MCP Bus URL for clients: MCP_BUS_URL
- Agent URLs (override defaults if not localhost): CHIEF_URL, SCOUT_URL, FACT_URL, ANALYST_URL, SYNTH_URL, CRITIC_URL, MEMORY_URL
- Tracing (optional): OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME
- Models/caching (local-first): HF_TOKEN (bootstrap), TRANSFORMERS_CACHE, HF_HOME, TRANSFORMERS_OFFLINE=1 (post-bootstrap)
- Training data & checkpoints (persist/back up): JUSTNEWS_DATASETS_DIR, JUSTNEWS_CHECKPOINTS_DIR, JUSTNEWS_EXPORTS_DIR
