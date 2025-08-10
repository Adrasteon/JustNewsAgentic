-- JustNews V4 - Core PostgreSQL Schema
-- Safe to run multiple times (idempotent via IF NOT EXISTS)

CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Optional embedding storage for sentence-transformer vectors (no pgvector required)
ALTER TABLE IF EXISTS articles
    ADD COLUMN IF NOT EXISTS embedding REAL[];

CREATE TABLE IF NOT EXISTS training_examples (
    id SERIAL PRIMARY KEY,
    task TEXT NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    critique TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Basic index for vector-like queries on metadata keys commonly used
CREATE INDEX IF NOT EXISTS idx_articles_metadata ON articles USING GIN (metadata);

-- Optional view for quick counts
CREATE OR REPLACE VIEW v_counts AS
SELECT
  (SELECT COUNT(*) FROM articles) AS articles_count,
  (SELECT COUNT(*) FROM training_examples) AS training_examples_count;
