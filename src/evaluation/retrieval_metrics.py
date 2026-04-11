"""Retrieval evaluation metrics: Recall@K, Precision@K, MRR, NDCG.

All functions take:
  - retrieved: list of doc_ids returned by the retriever (ranked order)
  - relevant: set of ground-truth relevant doc_ids
"""

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents found in top-k."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of the first relevant result."""
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Average Precision — area under the precision-recall curve.

    AP = (1/|relevant|) * sum(precision@k * rel(k)) for k=1..N
    where rel(k) = 1 if result at rank k is relevant, else 0.
    """
    if not relevant:
        return 0.0

    hits = 0
    sum_precision = 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            sum_precision += hits / i

    return sum_precision / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Uses binary relevance: rel = 1 if relevant, 0 otherwise.
    DCG@k  = sum( rel(i) / log2(i+1) ) for i=1..k
    IDCG@k = sum( 1 / log2(i+1) ) for i=1..min(k, |relevant|)
    """
    top_k = retrieved[:k]

    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant
    )

    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_all(
    retrieved: list[str],
    relevant: set[str],
    k_values: list[int],
) -> dict[str, float]:
    """Compute all metrics at each k value. Returns a flat dict."""
    results = {"mrr": mrr(retrieved, relevant)}

    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        results[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
        results[f"ap@{k}"] = average_precision(retrieved[:k], relevant)

    return results
