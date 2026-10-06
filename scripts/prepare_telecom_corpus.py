import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "talkmap/telecom-conversation-corpus"

OUTPUT_PATH = Path(
    "data/processed/telecom_historical_test.json"
)

MAX_CONVERSATIONS = 5000


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """Remove sensitive identifiers and normalize whitespace."""

    if not text:
        return ""

    text = str(text)

    # Email
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        text,
    )

    # URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "[URL]",
        text,
        flags=re.IGNORECASE,
    )

    # Phone numbers
    text = re.sub(
        r"\b(?:\+?\d[\d\s().-]{7,}\d)\b",
        "[PHONE]",
        text,
    )

    # Account/PIN identifiers
    text = re.sub(
        r"(?i)\b("
        r"pin|pincode|pin code|"
        r"account number|account no\.?|"
        r"customer number|customer id|"
        r"reference number|reference id"
        r")"
        r"\s*(?:is|:)?\s*[A-Za-z0-9#-]+\b",
        r"\1 [REDACTED]",
        text,
    )

    # Long numeric identifiers
    text = re.sub(
        r"\b\d{6,}\b",
        "[NUMBER]",
        text,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text: str) -> list[str]:
    """Split a conversation turn into lightweight sentences."""

    if not text:
        return []

    text = clean_text(text)

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if sentence.strip()
    ]


# ============================================================
# COMMON FILTERS
# ============================================================

IDENTITY_PHRASES = [
    "account pin",
    "pin is",
    "account number",
    "customer number",
    "customer id",
    "reference number",
    "reference id",
    "verify your identity",
    "verify my identity",
    "verification",
    "verify",
    "provide your account",
    "provide account",
]

CLOSING_PHRASES = [
    "thank you",
    "thanks",
    "okay",
    "ok",
    "alright",
    "got it",
    "i'll try",
    "i will try",
    "i'll do that",
    "i will do that",
    "sounds good",
    "sounds great",
    "that's great",
    "that is great",
    "bye",
    "goodbye",
    "have a great day",
    "is there anything else",
    "anything else",
    "what else can i do",
    "when do you expect",
    "when will it be fixed",
    "when will this be fixed",
]

PROBLEM_KEYWORDS = [
    # Network
    "network",
    "signal",
    "reception",
    "coverage",
    "no service",
    "no signal",
    "poor signal",
    "poor reception",
    "weak signal",
    "service outage",
    "network outage",
    "outage",

    # Calls
    "dropped call",
    "dropped calls",
    "call drop",
    "calls dropping",
    "can't make calls",
    "cannot make calls",
    "unable to make calls",
    "call quality",
    "poor call quality",

    # Data
    "data connection",
    "mobile data",
    "cellular data",
    "data isn't working",
    "data is not working",
    "data not working",
    "internet isn't working",
    "internet is not working",
    "internet not working",
    "slow data",
    "data is slow",
    "slow internet",
    "data connectivity",

    # SMS
    "sms",
    "text message",
    "text messages",
    "messages aren't working",
    "messages are not working",
    "can't send messages",
    "cannot send messages",

    # SIM / eSIM
    "sim card",
    "sim",
    "esim",

    # Roaming
    "roaming",
    "international roaming",

    # APN
    "apn",

    # Billing
    "billing",
    "bill",
    "charged",
    "charge",
    "incorrect charge",
    "wrong charge",
    "overcharged",

    # Recharge
    "recharge",
    "top up",
    "top-up",
    "recharge failed",

    # Plan
    "data plan",
    "subscription",
    "upgrade plan",
    "change plan",

    # Account access
    "account access",
    "login",
    "log in",
    "password",
    "locked out",
    "can't access",
    "cannot access",
]


# ============================================================
# PROBLEM HELPERS
# ============================================================

def contains_problem_keyword(text: str) -> bool:
    lower = text.lower()

    return any(
        keyword in lower
        for keyword in PROBLEM_KEYWORDS
    )


def is_identity_sentence(text: str) -> bool:
    lower = text.lower()

    return any(
        phrase in lower
        for phrase in IDENTITY_PHRASES
    )


def is_closing_sentence(text: str) -> bool:
    lower = text.lower().strip()

    return any(
        phrase in lower
        for phrase in CLOSING_PHRASES
    )


def is_explicit_problem_sentence(text: str) -> bool:
    """
    True only when the sentence itself describes the support
    problem, rather than merely mentioning a troubleshooting
    action.
    """

    if not text:
        return False

    lower = text.lower()

    if is_identity_sentence(text):
        return False

    if is_closing_sentence(text):
        return False

    # These indicate actions already attempted rather than
    # the underlying problem.
    troubleshooting_only = [
        "i've tried",
        "i have tried",
        "i tried",
        "we tried",
        "restarting my phone",
        "restart my phone",
        "checking for network outages",
        "checked for network outages",
        "software update",
        "software updates",
    ]

    if any(
        phrase in lower
        for phrase in troubleshooting_only
    ):
        return False

    return contains_problem_keyword(text)


# ============================================================
# AGENT-DESCRIBED PROBLEM
# ============================================================

def is_agent_problem_statement(text: str) -> bool:
    """
    Detect when the agent explicitly restates the customer's
    underlying problem.

    This is important for conversations such as Conversation 7,
    where the customer initially gives account information and
    the agent describes the problem before the customer adds
    details.
    """

    if not text:
        return False

    lower = text.lower()

    if is_identity_sentence(text):
        return False

    if not contains_problem_keyword(text):
        return False

    indicators = [
        "you're experiencing",
        "you are experiencing",
        "you're having",
        "you are having",
        "issues with your",
        "problems with your",
        "dropped calls",
        "poor reception",
        "data connectivity",
        "the issue",
        "the problem",
    ]

    return any(
        indicator in lower
        for indicator in indicators
    )


# ============================================================
# CUSTOMER PROBLEM EXTRACTION
# ============================================================

def extract_customer_problem(rows: list[dict]) -> str:
    """
    Extract the actual support problem.

    Strategy:

    1. Find the earliest explicit customer problem.
    2. Add only directly related customer context.
    3. Do not add troubleshooting attempts.
    4. Do not add identity verification.
    5. Do not add later sales/device-purchase dialogue.
    6. If no explicit customer problem exists, use the earliest
       agent statement that clearly restates the problem.
    """

    customer_sentences = []

    for row in rows:

        if row["speaker"].lower() != "client":
            continue

        for sentence in split_sentences(row["text"]):

            sentence = clean_text(sentence)

            if sentence:
                customer_sentences.append(sentence)

    # --------------------------------------------------------
    # Find first explicit customer problem.
    # --------------------------------------------------------

    first_problem_index = None

    for index, sentence in enumerate(
        customer_sentences
    ):

        if is_explicit_problem_sentence(sentence):

            first_problem_index = index
            break

    selected = []

    # --------------------------------------------------------
    # If customer explicitly states the problem.
    # --------------------------------------------------------

    if first_problem_index is not None:

        selected.append(
            customer_sentences[first_problem_index]
        )

        # Only inspect the next few client statements.
        #
        # We deliberately do NOT scan the entire conversation,
        # because later messages may belong to troubleshooting,
        # closing or sales.
        #
        # This prevents Conversation 10 from acquiring:
        # "What kind of phones do you recommend?"
        # as part of the original ticket.
        # ----------------------------------------------------

        candidate_window = customer_sentences[
            first_problem_index + 1:
            first_problem_index + 5
        ]

        context_keywords = [
            "happening",
            "started",
            "days",
            "weeks",
            "everywhere",
            "indoors",
            "outdoors",
            "driving",
            "city",
            "area",
            "location",
            "home",
            "work",
            "office",
            "frustrating",
            "problem",
            "issue",
            "issues",
            "trouble",
            "dropped calls",
            "poor reception",
            "data connection",
            "data connectivity",
            "slow",
            "no service",
        ]

        for sentence in candidate_window:

            lower = sentence.lower()

            if is_identity_sentence(sentence):
                continue

            if is_closing_sentence(sentence):
                continue

            # Never treat troubleshooting actions as the
            # underlying customer problem.
            if any(
                phrase in lower
                for phrase in [
                    "i've tried",
                    "i have tried",
                    "i tried",
                    "i'll try",
                    "i will try",
                    "what kind of phones",
                    "what phones",
                    "recommend",
                    "upgrade",
                    "cart",
                    "pricing",
                ]
            ):
                continue

            if any(
                keyword in lower
                for keyword in context_keywords
            ):
                selected.append(sentence)

    # --------------------------------------------------------
    # If no explicit customer problem exists, use an agent
    # problem statement.
    #
    # Conversation 7 falls into this path because the initial
    # client turn contains account information, while the agent
    # explicitly describes the problem.
    # --------------------------------------------------------

    if not selected:

        for row in rows:

            if row["speaker"].lower() != "agent":
                continue

            for sentence in split_sentences(row["text"]):

                sentence = clean_text(sentence)

                if is_agent_problem_statement(sentence):

                    selected.append(sentence)
                    break

            if selected:
                break

        # Add the customer's next useful contextual statement.
        if selected:

            problem_index_found = False

            for row in rows:

                if row["speaker"].lower() != "client":
                    continue

                for sentence in split_sentences(
                    row["text"]
                ):

                    sentence = clean_text(sentence)

                    lower = sentence.lower()

                    if is_identity_sentence(sentence):
                        continue

                    if is_closing_sentence(sentence):
                        continue

                    if any(
                        phrase in lower
                        for phrase in [
                            "i've tried",
                            "i have tried",
                            "i tried",
                            "what kind of phones",
                            "what phones",
                            "upgrade",
                        ]
                    ):
                        continue

                    # The first useful client sentence after
                    # the problem has been identified.
                    if any(
                        keyword in lower
                        for keyword in [
                            "days",
                            "weeks",
                            "indoors",
                            "driving",
                            "city",
                            "area",
                            "work",
                            "frustrating",
                        ]
                    ):
                        selected.append(sentence)
                        problem_index_found = True
                        break

                if problem_index_found:
                    break

    # --------------------------------------------------------
    # Remove duplicates.
    # --------------------------------------------------------

    unique = []
    seen = set()

    for sentence in selected:

        normalized = re.sub(
            r"\s+",
            " ",
            sentence.lower().strip(),
        )

        if normalized not in seen:

            seen.add(normalized)
            unique.append(sentence)

    return " ".join(unique[:4])


# ============================================================
# RESOLUTION KEYWORDS
# ============================================================

RESOLUTION_KEYWORDS = [
    # Troubleshooting
    "restart",
    "reboot",
    "reset",
    "network settings",
    "software update",
    "software updates",
    "update your phone",
    "network mode",
    "airplane mode",
    "turn off",
    "turn on",

    # Diagnosis
    "network issue",
    "network problem",
    "network congestion",
    "network outage",
    "coverage issue",
    "signal issue",
    "maintenance",
    "diagnostic test",
    "check the status",
    "checked on our network",
    "checked on your account",
    "known issues",

    # Escalation
    "escalate",
    "escalated",
    "escalation",
    "engineering team",
    "engineers",
    "technician",
    "support line",
    "dedicated support",
    "further assistance",

    # Support remedies
    "network booster",
    "data boost",
    "workaround",
    "discount",
    "refund",
    "credit",

    # Resolution
    "resolve",
    "resolved",
    "fix",
    "fixed",
    "solution",
    "recommend",
    "recommendation",
]


# ============================================================
# RESOLUTION EXTRACTION
# ============================================================

def is_identity_failure_sentence(text: str) -> bool:
    """
    True only when identity verification itself becomes
    the final support outcome.
    """

    lower = text.lower()

    failure_phrases = [
        "unable to verify",
        "unable to access your account",
        "unable to assist you further",
        "provide any further assistance",
        "doesn't match",
        "does not match",
    ]

    return any(
        phrase in lower
        for phrase in failure_phrases
    )


def is_useful_resolution_sentence(
    text: str,
    identity_failure: bool,
) -> bool:
    """
    Determine whether an agent sentence is useful historical
    resolution evidence.
    """

    if not text:
        return False

    lower = text.lower().strip()

    # Identity dialogue is normally excluded.
    if is_identity_sentence(text):

        if identity_failure:
            return is_identity_failure_sentence(text)

        return False

    # Remove filler / closing / sales dialogue.
    excluded = [
        "hello",
        "good morning",
        "good afternoon",
        "good evening",
        "thank you",
        "thanks for",
        "one moment",
        "just moment",
        "please hold",
        "bear with me",
        "i'll do my best",
        "i will do my best",
        "would you like",
        "feel free to",
        "is there anything else",
        "have a great day",
        "goodbye",
        "survey",
        "feedback",
        "what kind of phones",
        "which phones",
        "available for upgrade",
        "add it to your cart",
        "pricing",
        "promotion",
        "new line",
    ]

    if any(
        phrase in lower
        for phrase in excluded
    ):
        return False

    return any(
        keyword in lower
        for keyword in RESOLUTION_KEYWORDS
    )


def extract_agent_resolution(rows: list[dict]) -> str:
    """
    Extract diagnosis, troubleshooting and escalation.

    Identity verification is included only when it is the
    actual reason the ticket could not be resolved.
    """

    agent_text = " ".join(
        row["text"].lower()
        for row in rows
        if row["speaker"].lower() == "agent"
    )

    identity_failure = any(
        phrase in agent_text
        for phrase in [
            "unable to verify",
            "unable to access your account",
            "unable to assist you further",
            "provide any further assistance",
            "doesn't match",
            "does not match",
        ]
    )

    selected = []

    for row in rows:

        if row["speaker"].lower() != "agent":
            continue

        for sentence in split_sentences(row["text"]):

            sentence = clean_text(sentence)

            if not sentence:
                continue

            if is_useful_resolution_sentence(
                sentence,
                identity_failure,
            ):
                selected.append(sentence)

    # --------------------------------------------------------
    # Remove duplicates.
    # --------------------------------------------------------

    unique = []
    seen = set()

    for sentence in selected:

        normalized = re.sub(
            r"\s+",
            " ",
            sentence.lower().strip(),
        )

        if normalized not in seen:

            seen.add(normalized)
            unique.append(sentence)

    return " ".join(unique[:10])


# ============================================================
# RECORD CREATION
# ============================================================

def create_record(
    conversation_id: str,
    rows: list[dict],
) -> dict | None:

    customer_problem = extract_customer_problem(
        rows
    )

    historical_answer = extract_agent_resolution(
        rows
    )

    if not customer_problem:
        return None

    if not historical_answer:

        historical_answer = (
            "No clear resolution was provided in the "
            "historical conversation. Further investigation "
            "or escalation may be required."
        )

    title = customer_problem[:100]

    if len(customer_problem) > 100:
        title += "..."

    hash_input = (
        customer_problem
        + "\n"
        + historical_answer
    )

    content_hash = hashlib.sha256(
        hash_input.encode("utf-8")
    ).hexdigest()

    return {
        "source_id": f"CONV-{conversation_id}",
        "source_type": "historical_ticket",
        "title": title,
        "content": customer_problem,
        "product": None,
        "intent": None,
        "severity": None,
        "metadata": {
            "historical_answer": historical_answer,
            "conversation_id": conversation_id,
            "speaker_types": [
                "client",
                "agent",
            ],
        },
        "content_hash": content_hash,
    }


# ============================================================
# DATASET LOADING
# ============================================================

def load_conversations(
    max_conversations: int,
) -> list[tuple[str, list[dict]]]:

    print(
        f"Loading dataset: {DATASET_NAME}"
    )

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    conversations = {}

    for row in dataset:

        conversation_id = str(
            row["conversation_id"]
        )

        if conversation_id not in conversations:

            if len(conversations) >= max_conversations:
                break

            conversations[conversation_id] = []

        conversations[conversation_id].append(
            {
                "conversation_id": conversation_id,
                "speaker": str(
                    row["speaker"]
                ),
                "text": str(
                    row["text"]
                ),
            }
        )

    return list(
        conversations.items()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "TELECOM HISTORICAL CORPUS PREPROCESSING"
    )
    print("=" * 70)

    print(
        f"\nDataset: {DATASET_NAME}"
    )

    print(
        f"Target conversations: {MAX_CONVERSATIONS}"
    )

    conversations = load_conversations(
        MAX_CONVERSATIONS
    )

    print(
        f"\nLoaded {len(conversations)} conversations."
    )

    records = []
    skipped = 0

    for index, (
        conversation_id,
        rows,
    ) in enumerate(
        conversations,
        start=1,
    ):

        record = create_record(
            conversation_id,
            rows,
        )

        if record is None:

            skipped += 1

            print(
                f"[{index}/{len(conversations)}] "
                f"{conversation_id} -> SKIPPED"
            )

            continue

        records.append(record)

        print(
            f"[{index}/{len(conversations)}] "
            f"{record['source_id']}"
        )

        print("  Problem:")
        print(
            f"    {record['content']}"
        )

        print("  Resolution:")
        print(
            f"    {record['metadata']['historical_answer']}"
        )

        print()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Input conversations : {len(conversations)}"
    )

    print(
        f"Output records       : {len(records)}"
    )

    print(
        f"Skipped conversations: {skipped}"
    )

    print(
        f"Output file          : {OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()