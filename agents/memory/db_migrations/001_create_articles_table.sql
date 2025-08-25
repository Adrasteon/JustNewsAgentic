-- Drop dependent objects and the articles table
DROP TABLE IF EXISTS article_vectors CASCADE;
DROP TABLE IF EXISTS articles CASCADE;

-- Recreate the articles table without SERIAL (no sequence dependency)
CREATE TABLE articles (
    id INTEGER PRIMARY KEY DEFAULT 1,
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding NUMERIC[]
);

-- Recreate the article_vectors table with a foreign key to articles
-- Note: article_vectors is created in a later migration (003) using the pgvector VECTOR type.
-- This migration intentionally does not recreate article_vectors to avoid conflicts.
