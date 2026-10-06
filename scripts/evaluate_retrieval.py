import json
from pathlib import Path

from app.retrieval import hybrid_search
from app.analyzer import query_analyzer


EVAL_FILE = Path("data/evaluation/retrieval_eval.json")


def load_evaluation_data():
    with open(EVAL_FILE, "r") as f:
        return json.load(f)


def evaluate_query(item):
    query = item["query"]
    relevant_sources = set(item["relevant_sources"])

    analysis = query_analyzer.analyze(query)

    results = hybrid_search(
        query,
        limit=5,
        analysis=analysis,
    )

    retrieved_sources = [
        result["source_id"]
        for result in results
    ]

    # Recall@5:
    # 1 if at least one relevant source appears in top 5.
    hit = any(
        source_id in relevant_sources
        for source_id in retrieved_sources
    )

    recall_at_5 = 1.0 if hit else 0.0

    # MRR@5:
    # Reciprocal rank of the first relevant result.
    mrr_at_5 = 0.0

    for rank, source_id in enumerate(
        retrieved_sources,
        start=1,
    ):
        if source_id in relevant_sources:
            mrr_at_5 = 1.0 / rank
            break

    return {
        "query": query,
        "expected": list(relevant_sources),
        "retrieved": retrieved_sources,
        "recall_at_5": recall_at_5,
        "mrr_at_5": mrr_at_5,
    }


def main():
    evaluation_data = load_evaluation_data()

    results = []

    print("=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)

    for index, item in enumerate(
        evaluation_data,
        start=1,
    ):
        result = evaluate_query(item)
        results.append(result)

        status = "PASS" if result["recall_at_5"] == 1.0 else "FAIL"

        print(
            f"[{index:02d}/"
            f"{len(evaluation_data):02d}] "
            f"{status} | "
            f"MRR@5={result['mrr_at_5']:.3f} | "
            f"{result['query']}"
        )

    total = len(results)

    recall_at_5 = (
        sum(r["recall_at_5"] for r in results)
        / total
    )

    mrr_at_5 = (
        sum(r["mrr_at_5"] for r in results)
        / total
    )

    failed = [
        r for r in results
        if r["recall_at_5"] == 0.0
    ]

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"Queries evaluated : {total}")
    print(f"Recall@5          : {recall_at_5:.4f}")
    print(f"Recall@5 (%)      : {recall_at_5 * 100:.2f}%")
    print(f"MRR@5             : {mrr_at_5:.4f}")
    print(f"MRR@5 (%)         : {mrr_at_5 * 100:.2f}%")
    print(f"Failed queries    : {len(failed)}")

    if failed:
        print("\n" + "=" * 80)
        print("FAILED QUERIES")
        print("=" * 80)

        for result in failed:
            print(f"\nQuery: {result['query']}")
            print(
                "Expected:",
                ", ".join(result["expected"]),
            )
            print(
                "Retrieved:",
                ", ".join(result["retrieved"]),
            )

    # Save detailed results.
    output_file = Path(
        "data/evaluation/retrieval_results.json"
    )

    with open(output_file, "w") as f:
        json.dump(
            {
                "num_queries": total,
                "recall_at_5": recall_at_5,
                "mrr_at_5": mrr_at_5,
                "failed_queries": len(failed),
                "results": results,
            },
            f,
            indent=2,
        )

    print(
        f"\nDetailed results saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()
