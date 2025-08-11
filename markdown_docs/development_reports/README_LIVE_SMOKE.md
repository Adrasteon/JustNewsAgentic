# Live Smoke Test (JustNews V4)

This is a minimal end-to-end verifier that touches the real services, optional live crawling, and the Memory DB.

What it does:
- Ensures Playwright/Chromium are available for crawlers (installs if missing)
- Probes health of MCP Bus and agents
- Confirms Memory readiness (DB + pgvector)
- Optionally runs the BBC crawler for a short pass
- Exercises Memory endpoints `/save_article`, `/get_article/{id}`, and `/vector_search_articles`

## Prereqs
- Services running via systemd scripts under `deploy/systemd/` (no Docker)
- PostgreSQL reachable; `vector` extension enabled
- Shared secret set (recommended): `export MCP_SERVICE_TOKEN=...` in the service environment(s)

## Quick start
Run from repo root in your Python environment:

```bash
python scripts/live_smoke.py
```

Optional overrides when your services run on different hosts/ports:

```bash
MCP_BUS_URL=http://bus:8000 \
CHIEF_URL=http://chief:8001 \
SCOUT_URL=http://scout:8002 \
FACT_URL=http://fact:8003 \
ANALYST_URL=http://analyst:8004 \
SYNTH_URL=http://synth:8005 \
CRITIC_URL=http://critic:8006 \
MEMORY_URL=http://memory:8007 \
python scripts/live_smoke.py
```

## Expected output
- `[ok]` lines for each service health
- Memory readiness summary: `ready=true, db_ok=true, vector_ok=true`
- `PASS` at the end on success

If something fails, the script prints a concise reason and returns a non‑zero exit code.
