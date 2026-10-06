import json
import re
from pathlib import Path


INPUT_PATH = Path("data/processed/telecom_historical_test.json")
OUTPUT_PATH = Path("data/processed/telecom_historical_clean.json")


SALES_WORDS = [
    "discount",
    "new device",
    "upgrade",
    "new phone",
    "premium plan",
    "data plan",
]

TROUBLESHOOTING_WORDS = [
    "restart",
    "reset",
    "check",
    "network",
    "outage",
    "maintenance",
    "wi-fi",
    "wifi",
    "apn",
    "settings",
    "engineering",
    "technician",
    "diagnostic",
    "signal",
    "reception",
    "calling",
    "calls",
    "data",
    "coverage",
    "service",
    "resolve",
    "issue",
    "problem",
    "troubleshoot",
    "escalate",
]

# Obvious transcription corruption seen in the corpus.
REPLACEMENTS = {
    "let'm": "let me",
    "Let'm": "Let me",
    "recommendd": "recommend",
    "recommend like": "recommend",
    "engineeringers": "engineers",
    "compl solution": "complete solution",
    "compl solutions": "complete solutions",
    "solution solution": "solution",
    "poorreception": "poor reception",
    "bygoing": "by going",
    "networksettings": "network settings",
    "asa gesture": "as a gesture",
    "checking tosoftware": "checking for software",
    "still't": "still doesn't",
    "there issue": "the issue",
    "Would you tried": "Would you try",
    "However,I": "However, I",
}

# Transcript stage directions / speaker annotations.
STAGE_DIRECTION_PATTERN = re.compile(
    r"\([^)]{1,80}\)"
)


def clean_text(text: str) -> str:
    """Clean obvious transcription artifacts without rewriting the content."""
    if not text:
        return ""

    text = text.strip()

    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)

    # Remove transcript annotations such as:
    # (pause), (sighing), (interrupting), (apathetic), etc.
    text = STAGE_DIRECTION_PATTERN.sub("", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_sales_only(resolution: str) -> bool:
    """
    Remove records where the extracted resolution is essentially
    a commercial offer rather than a troubleshooting/resolution action.
    """
    text = resolution.lower()

    has_sales = any(
        word in text
        for word in SALES_WORDS
    )

    has_troubleshooting = any(
        word in text
        for word in TROUBLESHOOTING_WORDS
    )

    return has_sales and not has_troubleshooting


def normalize_for_duplicate(text: str) -> str:
    """Normalize text for exact duplicate detection."""
    text = clean_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    print("=" * 70)
    print("TELECOM CORPUS CLEANING")
    print("=" * 70)

    print("Input records:", len(records))

    cleaned = []

    removed_sales = 0
    removed_short = 0
    removed_duplicates = 0
    cleaned_artifacts = 0

    seen_pairs = set()

    for record in records:
        title = record.get("title", "").strip()

        metadata = record.get("metadata", {})
        resolution = metadata.get("historical_answer", "").strip()

        if not title or not resolution:
            removed_short += 1
            continue

        original_text = title + " " + resolution

        # --------------------------------------------------
        # Clean obvious transcription artifacts
        # --------------------------------------------------

        title = clean_text(title)
        resolution = clean_text(resolution)

        if title + " " + resolution != original_text:
            cleaned_artifacts += 1

        # --------------------------------------------------
        # Remove sales-only records
        # --------------------------------------------------

        if is_sales_only(resolution):
            removed_sales += 1
            continue

        # --------------------------------------------------
        # Remove genuinely short records
        # --------------------------------------------------

        if len(title.split()) < 5:
            removed_short += 1
            continue

        if len(resolution.split()) < 8:
            removed_short += 1
            continue

        # --------------------------------------------------
        # Deduplicate identical problem + resolution pairs
        # --------------------------------------------------

        duplicate_key = (
            normalize_for_duplicate(title),
            normalize_for_duplicate(resolution),
        )

        if duplicate_key in seen_pairs:
            removed_duplicates += 1
            continue

        seen_pairs.add(duplicate_key)

        # --------------------------------------------------
        # Update record
        # --------------------------------------------------

        record["title"] = title
        record["content"] = title

        record["metadata"]["historical_answer"] = resolution

        cleaned.append(record)

    # ------------------------------------------------------
    # Save cleaned corpus
    # ------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            cleaned,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print("Input records:", len(records))
    print("Sales-only removed:", removed_sales)
    print("Short/empty removed:", removed_short)
    print("Exact duplicate pairs removed:", removed_duplicates)
    print("Records with text cleanup:", cleaned_artifacts)
    print("Final records:", len(cleaned))
    print("Output:", OUTPUT_PATH)

    print("=" * 70)

    # ------------------------------------------------------
    # Show a few cleaned examples
    # ------------------------------------------------------

    print("\nSAMPLE CLEANED RECORDS")

    for i, record in enumerate(cleaned[:5], start=1):
        print(f"\n[{i}] {record['source_id']}")
        print("Problem:")
        print(record["title"])
        print("Resolution:")
        print(record["metadata"]["historical_answer"])


if __name__ == "__main__":
    main()
