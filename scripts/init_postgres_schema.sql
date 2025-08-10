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

-- Ensure sequences/defaults exist and align for id columns
DO $$
BEGIN
    -- Articles sequence and default
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.sequences WHERE sequence_name = 'articles_id_seq'
    ) THEN
        CREATE SEQUENCE articles_id_seq;
    END IF;
    ALTER SEQUENCE articles_id_seq OWNED BY articles.id;
    ALTER TABLE articles ALTER COLUMN id SET DEFAULT nextval('articles_id_seq');
    PERFORM setval('articles_id_seq', COALESCE((SELECT MAX(id) FROM articles), 1), true);

    -- Training examples sequence and default (best-effort; ignore ownership issues)
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.sequences WHERE sequence_name = 'training_examples_id_seq'
        ) THEN
            CREATE SEQUENCE training_examples_id_seq;
        END IF;
        ALTER SEQUENCE training_examples_id_seq OWNED BY training_examples.id;
        ALTER TABLE training_examples ALTER COLUMN id SET DEFAULT nextval('training_examples_id_seq');
        PERFORM setval('training_examples_id_seq', COALESCE((SELECT MAX(id) FROM training_examples), 1), true);
    EXCEPTION WHEN OTHERS THEN
        -- Do not fail schema init if we don't own existing sequence
        NULL;
    END;
END $$;
