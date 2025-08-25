# End-to-end (E2E) test architecture and artifacts

This document describes the JustNewsAgentic E2E test flow, the artifacts produced during runs, and the orchestrator fallback that guarantees per-agent forensic exit files.

## Overview

E2E runs in this repository exercise the full stack: an ephemeral PostgreSQL (used for vector storage and pgvector extension), the MCP bus, and the set of agents (Scout, Analyst, Memory, Synthesizer, Fact Checker, Critic, Chief Editor, Reasoning, Newsreader, Balancer).

The standard orchestrator flow used by developers and CI is:

1. `./scripts/run_e2e_safe.sh --ci` (wrapper) → calls `scripts/live_run.sh`
2. `live_run.sh` starts ephemeral Postgres via docker-compose, creates a temporary test DB, and runs idempotent migrations (ensuring pgvector exists)
3. `live_run.sh` starts the orchestrator `scripts/start_all.sh` in the background
4. `start_all.sh` launches agents with `uvicorn` via `conda run --name justnews-v2-prod uvicorn ...` and redirects each agent's stdout/stderr to `logs/<agent>.log`
5. `start_all.sh` waits for agent `/health` (and `/ready` where applicable), writes immediate startup snapshots, runs integration tests (`pytest -m integration`) or the live validator script, and finally requests gradual shutdown

## Key artifacts produced during an E2E run

All artifacts are stored in `logs/` for easy collection and archiving by CI.

- `logs/agent_startup_<ts>.csv`: a CSV snapshot of agents, configured port, pid (detected at start), and logfile path captured immediately after startup
- `logs/ss_after_start_<ts>.txt`: output of `ss -ltnp` capturing sockets/listeners
- `logs/ps_after_start_<ts>.txt`: a `pgrep -a uvicorn` snapshot showing uvicorn processes
- `logs/uvicorn_pids_<ts>.txt`: mapping of PIDs detected by the orchestrator to listening ports
- `logs/shutdown_traces.log`: append-only JSONL capturing shutdown audit POSTs (received by agents' `/shutdown_audit`) and internal shutdown traces written by agents when available
- `logs/validator_run_<ts>.log`: output from the live validator which probes agents via MCP `/agents` and `/info` and runs functional checks
- `logs/<agent>.log`: per-agent stdout/stderr captured when agents are started by `start_all.sh`
- `logs/<agent>_exit_<ts>.json`: per-agent forensic exit file with fields {ts, agent, reason, pid, tail}. These may be produced by the agent itself (best-effort) or — guaranteed by the orchestrator fallback — created by `start_all.sh` if the agent did not produce one.
- `logs/e2e_failure_<ts>/`: archived artifacts collected by `live_run.sh` in case of validator failure

## Orchestrator fallback for exit files (why and how)

Background: registering on_shutdown and SIGTERM handlers in agents is a best-effort way for an agent to create an authoritative exit file. However, in practice, some server/workflow configurations (uvicorn worker models, process supervisors) can cause lifecycle hooks to run in contexts where writing the expected files is unreliable.

To make CI deterministic and resilient, `scripts/start_all.sh` now contains a fallback step performed after shutdown/teardown:

- After requesting graceful shutdowns and killing processes, the orchestrator checks `logs/` for `*_exit_*.json` files for each agent.
- For any agent without an exit file, the orchestrator creates `logs/<agent>_exit_<ts>.json` itself by tailing `logs/<agent>.log` and composing a small JSON object with: ts, agent, reason (`orchestrator_fallback`), pid (0 if not available), and `tail` (an array of the last N lines).

This guarantees that every E2E run yields a per-agent forensic artifact even when in-process handlers don't run.

## How to interpret exit artifacts

- `reason`:
  - `fastapi_on_shutdown` or `signal_<n>` — written by agents if their handlers ran successfully.
  - `orchestrator_fallback` — written by the orchestrator when no agent exit file was found.
- `tail`: the last N lines of the agent log at the time the exit file was written. This is the primary forensic data for understanding crashes or orchestrator-initiated shutdowns.
- `ts`: POSIX timestamp useful for correlating with `agent_startup` and `shutdown_traces.log` entries.

## Troubleshooting

- If an agent consistently fails to produce an agent-side exit file but the fallback exists, investigate the agent runtime model. Some common causes:
  - Uvicorn is running with a process model where the main process spawns worker subprocesses and lifecycle hooks are installed in a different process.
  - The agent is being terminated with SIGKILL or by an external process before handlers run.
- Use the `agent_startup` CSV and `uvicorn_pids` mapping to correlate which PID owned a port at startup. Combine with `shutdown_traces.log` to see whether `/shutdown_audit` calls were received.

## Future work (optional)

- Improve PID mapping in fallback exit files by reading the `uvicorn_pids` snapshot produced at startup and including the last-known pid for the agent when available.
- Add CI checks that insist every agent has a per-agent exit artifact and fail the run if neither agent-side nor orchestrator fallback artifacts exist.

---

Documentation created: 2025-08-25
