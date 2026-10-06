import hashlib
import json
from pathlib import Path

from app.database import get_connection
from app.embeddings import embedding_service


KB_PATH = Path("data/processed/telecom_kb.json")


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_kb() -> list[dict]:
    with KB_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def ingest_kb():
    articles = load_kb()

    if not articles:
        raise ValueError("KB file is empty.")

    texts = [article["content"] for article in articles]

    print(f"Loaded {len(articles)} KB articles.")
    print("Generating local BGE embeddings...")

    embeddings = embedding_service.embed_texts(texts)

    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Embedding dimension: {len(embeddings[0])}")

    with get_connection() as conn:
        with conn.cursor() as cur:
            for article, embedding in zip(articles, embeddings):
                content = article["content"]

                cur.execute(
                    """
                    INSERT INTO documents (
                        source_id,
                        source_type,
                        title,
                        content,
                        product,
                        intent,
                        severity,
                        metadata,
                        content_hash,
                        embedding
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (source_id)
                    DO UPDATE SET
                        source_type = EXCLUDED.source_type,
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        product = EXCLUDED.product,
                        intent = EXCLUDED.intent,
                        severity = EXCLUDED.severity,
                        metadata = EXCLUDED.metadata,
                        content_hash = EXCLUDED.content_hash,
                        embedding = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        article["source_id"],
                        article["source_type"],
                        article["title"],
                        content,
                        article.get("product"),
                        article.get("intent"),
                        article.get("severity"),
                        json.dumps(article.get("metadata", {})),
                        content_hash(content),
                        embedding,
                    ),
                )

        conn.commit()

    print(f"Successfully ingested {len(articles)} KB articles.")


if __name__ == "__main__":
    ingest_kb()