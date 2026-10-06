import time

from app.database import get_connection
from app.embeddings import embedding_service


BATCH_SIZE = 32


def main():
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, content
                FROM documents
                ORDER BY id
                """
            )

            documents = cur.fetchall()

        print(f"Documents found: {len(documents)}")
        print(f"Batch size: {BATCH_SIZE}")

        start_time = time.time()

        for start in range(0, len(documents), BATCH_SIZE):
            batch = documents[start:start + BATCH_SIZE]

            texts = [
                f"{title}\n\n{content}"
                for _, title, content in batch
            ]

            embeddings = embedding_service.embed_texts(texts)

            with conn.cursor() as cur:
                for (document_id, _, _), embedding in zip(
                    batch,
                    embeddings,
                ):
                    cur.execute(
                        """
                        UPDATE documents
                        SET embedding = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        (embedding, document_id),
                    )

            conn.commit()

            processed = min(
                start + BATCH_SIZE,
                len(documents),
            )

            print(
                f"Processed {processed}/{len(documents)}"
            )

        elapsed = time.time() - start_time

        print()
        print("✅ Embedding rebuild complete!")
        print(f"Documents processed: {len(documents)}")
        print(f"Time taken: {elapsed:.2f} seconds")

    finally:
        conn.close()


if __name__ == "__main__":
    main()