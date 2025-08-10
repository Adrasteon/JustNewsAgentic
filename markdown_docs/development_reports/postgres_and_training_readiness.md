# Postgres and Training Readiness

This note summarizes the minimal DB and metrics enhancements to support production readiness for Memory and training.

## What changed
- Memory Agent now exposes:
  - `GET /db/health` (always 200): attempts a short `SELECT version()` and reports `{available: bool, version?: str, error?: str}`
  - Basic connection pooling (psycopg2 SimpleConnectionPool) with fallback to direct connects
  - Prometheus metrics extended: `memory_db_health_checks_total`, `memory_db_last_available`
- A minimal PostgreSQL schema is provided: `scripts/init_postgres_schema.sql`
- All agents have `/warmup` and `/metrics` endpoints for operational visibility.

## How to initialize Postgres
Run the schema using psql (adjust env as needed):

```bash
psql "postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST/$POSTGRES_DB" -f scripts/init_postgres_schema.sql
```

## Operational checks
- `GET http://127.0.0.1:8007/db/health` → should return available=false when DB is down, true with version when up.
- `GET http://127.0.0.1:8007/metrics` → includes DB health metrics and warmups.

## Next steps
- Add DB connection retry/backoff strategy and circuit-breaker for write-heavy code paths.
- Add migrations/versions using Alembic if the schema evolves.
- Wire Prometheus to scrape all agents and alert on `*_db_last_available == 0` for prolonged durations.
