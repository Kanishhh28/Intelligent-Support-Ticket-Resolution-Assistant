import hashlib
import json
import random
import re

from datasets import load_dataset


DATASET_NAME = "Tobi-Bueck/customer-support-tickets"

RANDOM_SEED = 42

TOTAL_TICKETS = 5000
EVAL_TICKETS = 100


def mask_pii(text: str) -> str:
    """Mask common PII patterns in ticket text."""

    if not text:
        return ""

    # Email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        text,
    )

    # Phone numbers
    text = re.sub(
        r"\b(?:\+?\d[\d\s().-]{7,}\d)\b",
        "[PHONE]",
        text,
    )

    # URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "[URL]",
        text,
    )

    return text


def make_source_id(index: int) -> str:
    return f"TICKET-{index:06d}"


def make_content_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def main():
    print("Loading dataset...")

    dataset = load_dataset(DATASET_NAME)

    train = dataset["train"]

    # Keep English tickets only
    english = train.filter(
        lambda row: row["language"] == "en"
    )

    print(f"English tickets available: {len(english)}")

    # Convert to Python records
    records = [english[i] for i in range(len(english))]

    random.seed(RANDOM_SEED)
    random.shuffle(records)

    if len(records) < TOTAL_TICKETS + EVAL_TICKETS:
        raise ValueError(
            "Not enough English tickets for the requested split."
        )

    # Reserve evaluation tickets first
    eval_records = records[:EVAL_TICKETS]

    # Retrieval corpus
    corpus_records = records[
        EVAL_TICKETS:EVAL_TICKETS + TOTAL_TICKETS
    ]

    def transform(record, index):
        subject = (record.get("subject") or "").strip()
        body = (record.get("body") or "").strip()

        subject = mask_pii(subject)
        body = mask_pii(body)

        content = f"{subject}\n\n{body}".strip()

        tags = []

        for i in range(1, 9):
            tag = record.get(f"tag_{i}")

            if tag:
                tags.append(tag)

        metadata = {
            "queue": record.get("queue"),
            "historical_answer": mask_pii(
                record.get("answer") or ""
            ),
            "language": record.get("language"),
            "version": record.get("version"),
            "tags": tags,
        }

        return {
            "source_id": make_source_id(index),
            "source_type": "historical_ticket",
            "title": subject,
            "content": content,
            "product": None,
            "intent": record.get("type"),
            "severity": record.get("priority"),
            "metadata": metadata,
            "content_hash": make_content_hash(content),
        }

    processed_corpus = [
        transform(record, i + 1)
        for i, record in enumerate(corpus_records)
    ]

    processed_eval = [
        transform(record, 5001 + i)
        for i, record in enumerate(eval_records)
    ]

    corpus_path = "data/processed/tickets_5000.json"
    eval_path = "data/processed/eval_tickets_100.json"

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(
            processed_corpus,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(
            processed_eval,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("✅ Dataset preparation complete!")
    print(f"Retrieval corpus: {len(processed_corpus)}")
    print(f"Evaluation set:   {len(processed_eval)}")
    print()
    print(f"Corpus saved to: {corpus_path}")
    print(f"Evaluation saved to: {eval_path}")

    print()
    print("Sample processed record:")
    print(json.dumps(
        processed_corpus[0],
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()