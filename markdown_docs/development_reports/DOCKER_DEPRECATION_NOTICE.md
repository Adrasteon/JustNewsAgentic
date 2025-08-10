# Docker Deprecation Notice

As of Aug 10, 2025, Docker and Docker Compose are deprecated and removed from the JustNewsAgentic codebase. Deployment uses native systemd services under `deploy/systemd/`.

Highlights:

- All CI pipelines are Dockerless and focus on unit/static checks.
- Local and production runs use `deploy/systemd/enable_all.sh --fresh` and companion scripts.
- Any remaining Docker artifacts are archived under `archive_obsolete_files/` and must not be used.

Rationale:

- Reduced complexity and cold-start overhead
- Better GPU and filesystem control on production hosts
- Simplified observability and service supervision with systemd

See also: `deploy/systemd/` scripts and project README for current run instructions.# Docker Deprecation Notice (August 10, 2025)

Docker and docker-compose are deprecated for JustNewsAgentic. Production and development operations are standardized on native Ubuntu + systemd.

## What changed
- All services run as systemd instances via `deploy/systemd/justnews@.service`
- Startup ordering is enforced: MCP Bus first, then agents, using readiness gates
- Health and readiness checks use `/health` and `/ready` endpoints
- MCP Bus includes HTTP timeouts, retries, and a simple circuit breaker

## What to do instead
- Install units: `deploy/systemd/install_native.sh`
- Start all services (fresh): `deploy/systemd/enable_all.sh --fresh`
- Check health: `deploy/systemd/health_check.sh`
- Roll back: `deploy/systemd/rollback_native.sh`

## Files considered legacy
- `docker-compose.yml` (kept for archival reference)
- Agent-specific `Dockerfile` files under `agents/*/Dockerfile`
- `mcp_bus/Dockerfile`

These files must not be used for deployment going forward.

## Notes
- Warmup endpoints are available on heavy agents: `/warmup`
- Tests: run `pytest` for smoke endpoints and tools
- Documentation hub: `markdown_docs/README.md`
