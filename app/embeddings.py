from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-base-en-v1.5"


class EmbeddingService:
    def __init__(self):
        print(f"Loading local embedding model: {MODEL_NAME}")

        self.model = SentenceTransformer(MODEL_NAME)
        self.dimension = 768

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""

        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts locally."""

        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text.")

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        return embeddings.tolist()


embedding_service = EmbeddingService()