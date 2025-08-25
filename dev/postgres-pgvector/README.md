This directory contains a small Dockerfile to build a Postgres 15 image with the `pgvector` extension preinstalled.

How to rebuild locally:

```bash
# from repo root
docker-compose -f dev/docker-compose.postgres.yml build postgres
```

The resulting image will be tagged as `local/postgres:15-pgvector` and used by the compose file.

Note: the build pulls and compiles native code; it requires network access and may take a few minutes on first build.
