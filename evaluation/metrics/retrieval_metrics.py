"""
Retrieval quality metrics for evaluating the RAG retriever.

Implements standard IR metrics: Precision@K, Recall@K, Hit Rate@K,
MRR (Mean Reciprocal Rank), and NDCG@K.
"""
import math
from typing import List, Dict, Set, Tuple

from evaluation.eval_config import RETRIEVAL_K_VALUES, CHUNK_INDEX_TOLERANCE


def _is_relevant(
    retrieved_doc_id: str,
    retrieved_chunk_index: int,
    source_chunks: List[Dict],
    tolerance: int = CHUNK_INDEX_TOLERANCE,
) -> bool:
    """Check if a retrieved chunk is relevant to any source chunk.

    A retrieved chunk is considered relevant if it shares the same doc_id
    and its chunk_index is within +-tolerance of a source chunk's index.
    """
    for src in source_chunks:
        if retrieved_doc_id == src.get("doc_id", src.get("source", "")):
            src_idx = src.get("chunk_index", -999)
            if src_idx < 0:
                # No chunk index available — fall back to same-document match
                return True
            if abs(retrieved_chunk_index - src_idx) <= tolerance:
                return True
    return False


def _get_relevance_vector(
    retrieved_docs: List, source_chunks: List[Dict], tolerance: int = CHUNK_INDEX_TOLERANCE
) -> List[int]:
    """Return a binary relevance vector for the retrieved documents."""
    rel = []
    for doc in retrieved_docs:
        doc_id = doc.metadata.get("doc_id", doc.metadata.get("source", ""))
        chunk_idx = doc.metadata.get("chunk_index", -1)
        rel.append(1 if _is_relevant(doc_id, chunk_idx, source_chunks, tolerance) else 0)
    return rel


# ─── Per-query metrics ──────────────────────────────────────────────────────

def precision_at_k(retrieved_docs: List, source_chunks: List[Dict], k: int) -> float:
    """Fraction of top-K retrieved docs that are relevant."""
    if k == 0 or not retrieved_docs:
        return 0.0
    rel = _get_relevance_vector(retrieved_docs[:k], source_chunks)
    return sum(rel) / k


def recall_at_k(retrieved_docs: List, source_chunks: List[Dict], k: int) -> float:
    """Fraction of all source chunks that appear in top-K results."""
    if not source_chunks:
        return 1.0  # No expected chunks — trivially complete
    found = set()
    for doc in retrieved_docs[:k]:
        doc_id = doc.metadata.get("doc_id", doc.metadata.get("source", ""))
        chunk_idx = doc.metadata.get("chunk_index", -1)
        for i, src in enumerate(source_chunks):
            if _is_relevant(doc_id, chunk_idx, [src]):
                found.add(i)
    return len(found) / len(source_chunks)


def hit_rate_at_k(retrieved_docs: List, source_chunks: List[Dict], k: int) -> float:
    """Binary: 1.0 if at least one relevant doc appears in top-K, else 0.0."""
    rel = _get_relevance_vector(retrieved_docs[:k], source_chunks)
    return 1.0 if any(r == 1 for r in rel) else 0.0


def reciprocal_rank(retrieved_docs: List, source_chunks: List[Dict]) -> float:
    """1 / rank of the first relevant document. 0 if none found."""
    rel = _get_relevance_vector(retrieved_docs, source_chunks)
    for i, r in enumerate(rel):
        if r == 1:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_docs: List, source_chunks: List[Dict], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    rel = _get_relevance_vector(retrieved_docs[:k], source_chunks)

    # DCG
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))

    # Ideal DCG: all relevant docs at the top
    ideal_rel = sorted(rel, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rel))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ─── Aggregate across queries ───────────────────────────────────────────────

def compute_retrieval_metrics(
    results: List[Dict],
    k_values: List[int] = None,
) -> Dict:
    """Compute all retrieval metrics across a set of evaluation results.

    Args:
        results: List of dicts, each with:
            - "retrieved_docs": list of LangChain Document objects
            - "source_chunks": list of dicts with doc_id, chunk_index
        k_values: List of K values to evaluate at.

    Returns:
        Dict of metric_name -> value (averaged across queries).
    """
    if k_values is None:
        k_values = RETRIEVAL_K_VALUES

    metrics = {}
    n = len(results)
    if n == 0:
        return metrics

    for k in k_values:
        prec_scores = []
        rec_scores = []
        hit_scores = []
        ndcg_scores = []

        for r in results:
            docs = r["retrieved_docs"]
            src = r["source_chunks"]
            prec_scores.append(precision_at_k(docs, src, k))
            rec_scores.append(recall_at_k(docs, src, k))
            hit_scores.append(hit_rate_at_k(docs, src, k))
            ndcg_scores.append(ndcg_at_k(docs, src, k))

        metrics[f"precision@{k}"] = sum(prec_scores) / n
        metrics[f"recall@{k}"] = sum(rec_scores) / n
        metrics[f"hit_rate@{k}"] = sum(hit_scores) / n
        metrics[f"ndcg@{k}"] = sum(ndcg_scores) / n

    # MRR (not K-dependent)
    mrr_scores = [reciprocal_rank(r["retrieved_docs"], r["source_chunks"]) for r in results]
    metrics["mrr"] = sum(mrr_scores) / n

    return metrics


def compute_per_query_retrieval_metrics(
    retrieved_docs: List, source_chunks: List[Dict], k_values: List[int] = None
) -> Dict:
    """Compute retrieval metrics for a single query."""
    if k_values is None:
        k_values = RETRIEVAL_K_VALUES

    metrics = {}
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved_docs, source_chunks, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_docs, source_chunks, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_docs, source_chunks, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_docs, source_chunks, k)
    metrics["mrr"] = reciprocal_rank(retrieved_docs, source_chunks)
    return metrics
