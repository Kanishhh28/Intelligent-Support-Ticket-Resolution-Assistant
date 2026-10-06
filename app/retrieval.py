from collections import defaultdict

from pgvector import Vector

from app.database import get_connection
from app.embeddings import embedding_service


# =============================================================
# Retrieval configuration
# =============================================================

# Reciprocal Rank Fusion constant.
# Higher values make rank differences less aggressive.
RRF_K = 60

# Small metadata boosts applied after RRF.
PRODUCT_BOOST = 0.0020
INTENT_BOOST = 0.0030
SEVERITY_BOOST = 0.0005
KB_BOOST = 0.0200


# =============================================================
# Analyzer → Knowledge Base intent mapping
# =============================================================

INTENT_MAP = {
    "billing_issue": "billing dispute",
    "recharge_issue": "recharge failure",
    "mobile_data_not_working": "data not working",
    "slow_mobile_data": "slow data",
    "data_allowance_exhausted": "data allowance",
    "no_network_signal": "no signal",
    "call_drop": "call drops",
    "poor_call_quality": "call quality",
    "sms_not_working": "sms failure",
    "account_access": "account access",
    "roaming_issue": "roaming",
    "apn_issue": "apn configuration",
    "plan_issue": "plan activation",
    "esim_issue": "esim activation",
    "sim_issue": "sim not detected",

    # Newly added mappings
    "network_outage": "Network Outage",
    "network_degradation": "Network Degradation",
    "general_troubleshooting": "General Troubleshooting",
}


# =============================================================
# Metadata normalization helpers
# =============================================================

def _normalize_metadata_value(value):
    """
    Normalize metadata values so analyzer labels and
    database labels can be compared consistently.

    Examples:
        "network_outage"  -> "network outage"
        "Network Outage"  -> "network outage"
        "NETWORK OUTAGE"  -> "network outage"
    """
    if value is None:
        return ""

    return str(value).strip().lower().replace("_", " ")


# =============================================================
# Semantic search
# =============================================================

def semantic_search(query: str, limit: int = 25):
    """
    Retrieve documents using vector similarity.

    Uses the BGE embedding model and pgvector cosine distance.
    """

    query_embedding = embedding_service.embed_text(query)
    query_vector = Vector(query_embedding)

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    source_id,
                    source_type,
                    title,
                    content,
                    product,
                    intent,
                    severity,
                    metadata
                FROM documents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (query_vector, limit),
            )

            return cur.fetchall()

    finally:
        conn.close()


# =============================================================
# Keyword search
# =============================================================

def keyword_search(query: str, limit: int = 25):
    """
    Retrieve documents using PostgreSQL full-text search.
    """

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    source_id,
                    source_type,
                    title,
                    content,
                    product,
                    intent,
                    severity,
                    metadata
                FROM documents
                WHERE search_vector @@ websearch_to_tsquery(
                    'english',
                    %s
                )
                ORDER BY ts_rank_cd(
                    search_vector,
                    websearch_to_tsquery('english', %s)
                ) DESC
                LIMIT %s;
                """,
                (query, query, limit),
            )

            return cur.fetchall()

    finally:
        conn.close()


# =============================================================
# Authoritative Knowledge Base metadata search
# =============================================================

def metadata_kb_search(
    analysis: dict | None = None,
    limit: int = 5,
):
    """
    Retrieve authoritative Knowledge Base documents using
    query-analysis metadata.

    This provides a deterministic candidate path for cases
    where semantic or keyword retrieval misses the correct KB.
    """

    analysis = analysis or {}

    query_product = analysis.get("product")
    query_intent = analysis.get("intent")

    mapped_intent = INTENT_MAP.get(
        query_intent,
        query_intent,
    )

    if (
        not query_product
        and not mapped_intent
    ):
        return []

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    source_id,
                    source_type,
                    title,
                    content,
                    product,
                    intent,
                    severity,
                    metadata
                FROM documents
                WHERE source_type = 'knowledge_base'
                  AND (
                      REPLACE(LOWER(product), '_', ' ')
                          = REPLACE(LOWER(CAST(%s AS TEXT)), '_', ' ')
                      OR
                      REPLACE(LOWER(intent), '_', ' ')
                          = REPLACE(LOWER(CAST(%s AS TEXT)), '_', ' ')
                  )
                ORDER BY
                    CASE
                        WHEN
                            REPLACE(LOWER(product), '_', ' ')
                                = REPLACE(LOWER(CAST(%s AS TEXT)), '_', ' ')
                            AND
                            REPLACE(LOWER(intent), '_', ' ')
                                = REPLACE(LOWER(CAST(%s AS TEXT)), '_', ' ')
                        THEN 0

                        WHEN
                            REPLACE(LOWER(intent), '_', ' ')
                                = REPLACE(LOWER(CAST(%s AS TEXT)), '_', ' ')
                        THEN 1

                        ELSE 2
                    END,
                    source_id
                LIMIT %s;
                """,
                (
                    query_product,
                    mapped_intent,
                    query_product,
                    mapped_intent,
                    mapped_intent,
                    limit,
                ),
            )

            return cur.fetchall()

    finally:
        conn.close()


# =============================================================
# Reciprocal Rank Fusion
# =============================================================

def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    k: int = RRF_K,
):
    """
    Combine semantic and keyword retrieval using
    Reciprocal Rank Fusion.

    RRF score:

        1 / (k + rank)
    """

    scores = defaultdict(float)
    documents = {}

    # Add semantic-search contribution.
    for rank, row in enumerate(
        semantic_results,
        start=1,
    ):
        source_id = row[0]

        scores[source_id] += 1 / (k + rank)
        documents[source_id] = row

    # Add keyword-search contribution.
    for rank, row in enumerate(
        keyword_results,
        start=1,
    ):
        source_id = row[0]

        scores[source_id] += 1 / (k + rank)
        documents[source_id] = row

    return scores, documents


# =============================================================
# Metadata-aware reranking
# =============================================================

def apply_metadata_boost(
    scores,
    documents,
    analysis: dict | None = None,
):
    """
    Apply metadata-based boosts after RRF.

    Metadata is used as a soft ranking signal rather than
    a hard filter so that relevant documents are not removed.
    """

    analysis = analysis or {}

    query_intent = analysis.get("intent")
    query_product = analysis.get("product")
    query_severity = analysis.get("severity")

    mapped_intent = INTENT_MAP.get(
        query_intent,
        query_intent,
    )

    boosted_scores = {}

    for source_id, base_score in scores.items():
        row = documents[source_id]

        document_product = row[4]
        document_intent = row[5]
        document_severity = row[6]

        boost = 0.0

        # -----------------------------------------------------
        # Authoritative Knowledge Base boost
        # -----------------------------------------------------

        if row[1] == "knowledge_base":
            boost += KB_BOOST

        # -----------------------------------------------------
        # Product match
        # -----------------------------------------------------

        if (
            query_product
            and query_product != "unknown"
            and document_product
            and _normalize_metadata_value(query_product)
            == _normalize_metadata_value(document_product)
        ):
            boost += PRODUCT_BOOST

        # -----------------------------------------------------
        # Intent match
        # -----------------------------------------------------

        if (
            query_intent
            and query_intent != "unknown"
            and mapped_intent
            and mapped_intent != "unknown"
            and document_intent
            and _normalize_metadata_value(mapped_intent)
            == _normalize_metadata_value(document_intent)
        ):
            boost += INTENT_BOOST

        # -----------------------------------------------------
        # Severity match
        # -----------------------------------------------------

        if (
            query_severity
            and query_severity != "unknown"
            and document_severity
            and _normalize_metadata_value(query_severity)
            == _normalize_metadata_value(document_severity)
        ):
            boost += SEVERITY_BOOST

        boosted_scores[source_id] = (
            base_score + boost
        )

    return boosted_scores


# =============================================================
# Build application-level results
# =============================================================

def build_results(
    scores,
    documents,
):
    """
    Convert database rows into the application-level
    retrieval result format.
    """

    ranked = sorted(
        documents.items(),
        key=lambda item: scores[item[0]],
        reverse=True,
    )

    results = []

    for source_id, row in ranked:
        metadata = row[7] or {}

        results.append(
            {
                "source_id": source_id,
                "source_type": row[1],
                "title": row[2],
                "content": row[3],
                "product": row[4],
                "intent": row[5],
                "severity": row[6],
                "metadata": metadata,
                "historical_answer": (
                    metadata.get("historical_answer")
                    if row[1] == "historical_ticket"
                    else None
                ),
                "rrf_score": scores[source_id],
            }
        )

    return results


# =============================================================
# Hybrid search
# =============================================================

def hybrid_search(
    query: str,
    limit: int = 5,
    analysis: dict | None = None,
):
    """
    Hybrid retrieval pipeline:

        Query
          ↓
        Semantic Search
          +
        Keyword Search
          +
        Metadata-based KB Candidate Search
          ↓
        Reciprocal Rank Fusion
          ↓
        Metadata Boost
          ↓
        Top N Results

    Semantic and keyword retrieval provide broad recall.

    The metadata KB search provides an authoritative candidate
    path when the semantic or keyword searches miss the correct
    Knowledge Base article.
    """

    # ---------------------------------------------------------
    # Retrieve broad candidates from semantic and keyword search
    # ---------------------------------------------------------

    candidate_limit = 100

    semantic_results = semantic_search(
        query,
        limit=candidate_limit,
    )

    keyword_results = keyword_search(
        query,
        limit=candidate_limit,
    )

    # ---------------------------------------------------------
    # Retrieve authoritative KB candidates using analysis
    # ---------------------------------------------------------

    metadata_results = metadata_kb_search(
        analysis=analysis,
        limit=5,
    )

    # ---------------------------------------------------------
    # Add metadata KB candidates to the semantic candidate set
    #
    # RRF itself accepts two lists, so metadata candidates are
    # appended to semantic candidates only if they are not
    # already present.
    # ---------------------------------------------------------

    existing_semantic_ids = {
        row[0]
        for row in semantic_results
    }

    for row in metadata_results:
        if row[0] not in existing_semantic_ids:
            semantic_results.append(row)

    # ---------------------------------------------------------
    # Combine retrieval strategies using RRF
    # ---------------------------------------------------------

    scores, documents = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
    )

    # ---------------------------------------------------------
    # Apply metadata-aware ranking
    # ---------------------------------------------------------

    scores = apply_metadata_boost(
        scores=scores,
        documents=documents,
        analysis=analysis,
    )

    # ---------------------------------------------------------
    # Build ranked results
    # ---------------------------------------------------------

    results = build_results(
        scores=scores,
        documents=documents,
    )

    # ---------------------------------------------------------
    # Return requested number of results
    # ---------------------------------------------------------

    return results[:limit]