import json
import time
from pathlib import Path

from google.genai import errors

from app.database import get_connection
from app.embeddings import embedding_service


DATA_PATH = Path("data/processed/tickets_5000.json")
BATCH_SIZE = 50
MAX_RETRIES = 5
DEFAULT_RETRY_SECONDS = 45


def load_records():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_existing_source_ids():
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_id
                FROM documents
                WHERE source_type = 'historical_ticket'
                """
            )

            return {row[0] for row in cur.fetchall()}

    finally:
        conn.close()


def generate_embeddings_with_retry(texts):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return embedding_service.embed_texts(texts)

        except errors.ClientError as exc:
            if exc.code != 429:
                raise

            wait_seconds = DEFAULT_RETRY_SECONDS

            error_text = str(exc)

            if "retryDelay" in error_text:
                print(
                    f"⚠️ Gemini quota reached. "
                    f"Waiting {wait_seconds} seconds before retry "
                    f"(attempt {attempt}/{MAX_RETRIES})..."
                )
            else:
                print(
                    f"⚠️ Gemini request rate-limited. "
                    f"Waiting {wait_seconds} seconds "
                    f"(attempt {attempt}/{MAX_RETRIES})..."
                )

            time.sleep(wait_seconds)

    raise RuntimeError(
        "Gemini embedding failed after maximum retry attempts."
    )


def ingest_records(records, batch_size=BATCH_SIZE):

    existing_ids = get_existing_source_ids()

    pending_records = [
        record
        for record in records
        if record["source_id"] not in existing_ids
    ]

    print(f"Already in database: {len(existing_ids)}")
    print(f"Records to ingest: {len(pending_records)}")

    if not pending_records:
        print("✅ All records are already ingested.")
        return

    conn = get_connection()

    try:
        for start in range(0, len(pending_records), batch_size):

            batch = pending_records[start:start + batch_size]

            texts = [
                f"{record['title']}\n\n{record['content']}"
                for record in batch
            ]

            print(
                f"\nEmbedding batch "
                f"{start + 1}-{start + len(batch)} "
                f"of {len(pending_records)}..."
            )

            embeddings = generate_embeddings_with_retry(texts)

            with conn.cursor() as cur:

                for record, embedding in zip(batch, embeddings):

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
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (source_id)
                        DO UPDATE SET
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
                            record["source_id"],
                            record["source_type"],
                            record["title"],
                            record["content"],
                            record["product"],
                            record["intent"],
                            record["severity"],
                            json.dumps(record["metadata"]),
                            record["content_hash"],
                            embedding,
                        ),
                    )

            conn.commit()

            processed = min(
                start + batch_size,
                len(pending_records)
            )

            print(
                f"✅ Stored {processed}/{len(pending_records)} "
                f"new records"
            )

    finally:
        conn.close()


def main(limit=5000):

    records = load_records()

    records = records[:limit]

    print(f"Loaded {len(records)} records.")
    print(f"Batch size: {BATCH_SIZE}")

    start_time = time.time()

    ingest_records(
        records,
        batch_size=BATCH_SIZE,
    )

    elapsed = time.time() - start_time

    print()
    print("✅ Ingestion complete!")
    print(f"Time taken: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main(limit=5000)