import json
from psycopg.types.json import Jsonb
from pathlib import Path

from pgvector import Vector

from app.database import get_connection
from app.embeddings import embedding_service


INPUT_PATH = Path(
    "data/processed/telecom_historical_clean.json"
)

BATCH_SIZE = 32


def load_records():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    records = load_records()

    print("=" * 70)
    print("TELECOM HISTORICAL TICKET INGESTION")
    print("=" * 70)
    print("Records to ingest:", len(records))
    print("Embedding model:", "BAAI/bge-base-en-v1.5")
    print("Embedding dimension:", embedding_service.dimension)
    print()

    conn = get_connection()

    inserted = 0
    skipped = 0

    try:
        with conn.cursor() as cur:

            for start in range(0, len(records), BATCH_SIZE):
                batch = records[start:start + BATCH_SIZE]

                texts = [
                    f"{record['title']}\n\n"
                    f"Resolution: "
                    f"{record['metadata']['historical_answer']}"
                    for record in batch
                ]

                embeddings = embedding_service.embed_texts(texts)

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
                        DO NOTHING;
                        """,
                        (
                            record["source_id"],
                            "historical_ticket",
                            record["title"],
                            record["content"],
                            record.get("product"),
                            record.get("intent"),
                            record.get("severity"),
                            Jsonb(record.get("metadata", {})),
                            record["content_hash"],
                            Vector(embedding),
                        ),
                    )

                    if cur.rowcount == 1:
                        inserted += 1
                    else:
                        skipped += 1

                conn.commit()

                processed = min(
                    start + BATCH_SIZE,
                    len(records)
                )

                print(
                    f"Processed {processed}/{len(records)} "
                    f"| inserted={inserted} "
                    f"| skipped={skipped}"
                )

        print()
        print("=" * 70)
        print("INGESTION COMPLETE")
        print("=" * 70)
        print("Input records:", len(records))
        print("Inserted:", inserted)
        print("Skipped:", skipped)

    except Exception:
        conn.rollback()
        print("\nERROR: Transaction rolled back.")
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
