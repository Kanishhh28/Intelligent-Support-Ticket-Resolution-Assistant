CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,

    source_id TEXT UNIQUE NOT NULL,
    source_type TEXT NOT NULL,

    title TEXT,
    content TEXT NOT NULL,

    product TEXT,
    intent TEXT,
    severity TEXT,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    content_hash TEXT,

    embedding vector(768),

    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector(
            'english',
            coalesce(title, '') || ' ' ||
            coalesce(content, '')
        )
    ) STORED,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_search_vector
ON documents
USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS idx_documents_embedding
ON documents
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_documents_product
ON documents (product);

CREATE INDEX IF NOT EXISTS idx_documents_intent
ON documents (intent);

CREATE INDEX IF NOT EXISTS idx_documents_source_type
ON documents (source_type);