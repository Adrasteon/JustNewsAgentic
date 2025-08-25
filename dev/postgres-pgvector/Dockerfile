FROM postgres:15

# Install build dependencies to compile pgvector
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
        postgresql-server-dev-15 \
        llvm clang pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Build and install pgvector
RUN set -eux; \
    cd /tmp; \
    git clone --depth 1 https://github.com/pgvector/pgvector.git; \
    cd pgvector; \
    make && make install; \
    rm -rf /tmp/pgvector

# Keep default postgres entrypoint/command
EXPOSE 5432
