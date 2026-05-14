"""
Custom metrics for evaluating Cryostat RAG-specific features.

Tests the unique capabilities: dual-DB routing, context expansion,
vague query detection, follow-up reformulation, and document type filtering.
"""
import os
import sys
from typing import List, Dict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import evaluation.env_setup  # noqa: F401 — must precede rag imports

from rag import (
    ContextExpandingHybridRetriever,
    needs_clarification,
    is_follow_up_query,
    ConversationHistory,
    EXPAND_CONTEXT_BEFORE,
    EXPAND_CONTEXT_AFTER,
)


def dual_db_routing_accuracy(test_queries: List[Dict], retriever: ContextExpandingHybridRetriever) -> Dict:
    """Measure whether queries are routed to the correct database.

    Each test query should have an "expected_db" field ("fine" or "coarse").
    We check if the retriever would use the correct database based on
    the procedural keyword detection logic.
    """
    procedural_keywords = ["how", "procedure", "steps", "process", "install", "setup"]
    correct = 0
    total = 0
    per_category = {"fine_correct": 0, "fine_total": 0, "coarse_correct": 0, "coarse_total": 0}

    for q in test_queries:
        expected = q.get("expected_db")
        if expected not in ("fine", "coarse"):
            continue

        query_lower = q["question"].lower()
        is_procedural = any(k in query_lower for k in procedural_keywords)
        predicted = "coarse" if is_procedural else "fine"

        total += 1
        if predicted == expected:
            correct += 1

        if expected == "fine":
            per_category["fine_total"] += 1
            if predicted == "fine":
                per_category["fine_correct"] += 1
        else:
            per_category["coarse_total"] += 1
            if predicted == "coarse":
                per_category["coarse_correct"] += 1

    return {
        "routing_accuracy": correct / total if total > 0 else 0.0,
        "fine_accuracy": per_category["fine_correct"] / per_category["fine_total"] if per_category["fine_total"] > 0 else 0.0,
        "coarse_accuracy": per_category["coarse_correct"] / per_category["coarse_total"] if per_category["coarse_total"] > 0 else 0.0,
        "total_queries": total,
    }


def context_expansion_impact(
    test_queries: List[Dict],
    retriever: ContextExpandingHybridRetriever,
) -> Dict:
    """Compare retrieval with and without context expansion.

    Measures how much additional relevant content is captured by
    including neighboring chunks.
    """
    import rag

    expanded_counts = []
    unexpanded_counts = []
    expansion_ratios = []

    for q in test_queries:
        query = q["question"]

        # With expansion (default behavior)
        docs_expanded = retriever.get_relevant_documents(query)
        expanded_counts.append(len(docs_expanded))

        # Without expansion
        saved_before = rag.EXPAND_CONTEXT_BEFORE
        saved_after = rag.EXPAND_CONTEXT_AFTER
        rag.EXPAND_CONTEXT_BEFORE = 0
        rag.EXPAND_CONTEXT_AFTER = 0
        try:
            docs_unexpanded = retriever.get_relevant_documents(query)
        finally:
            rag.EXPAND_CONTEXT_BEFORE = saved_before
            rag.EXPAND_CONTEXT_AFTER = saved_after

        unexpanded_counts.append(len(docs_unexpanded))
        if len(docs_unexpanded) > 0:
            expansion_ratios.append(len(docs_expanded) / len(docs_unexpanded))

    return {
        "mean_expanded_chunks": float(np.mean(expanded_counts)) if expanded_counts else 0.0,
        "mean_unexpanded_chunks": float(np.mean(unexpanded_counts)) if unexpanded_counts else 0.0,
        "mean_expansion_ratio": float(np.mean(expansion_ratios)) if expansion_ratios else 0.0,
        "total_queries": len(test_queries),
    }


def vague_detection_accuracy(test_queries: List[Dict]) -> Dict:
    """Evaluate accuracy of the vague query detection system.

    Each test query should have an "expected_behavior" field.
    Queries with "should_clarify" should trigger needs_clarification().
    """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for q in test_queries:
        expected = q.get("expected_behavior", "")
        should_be_vague = expected == "should_clarify"
        detected_vague = needs_clarification(q["question"])

        if should_be_vague and detected_vague:
            true_positives += 1
        elif not should_be_vague and not detected_vague:
            true_negatives += 1
        elif not should_be_vague and detected_vague:
            false_positives += 1
        else:
            false_negatives += 1

    total = true_positives + true_negatives + false_positives + false_negatives
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": (true_positives + true_negatives) / total if total > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def follow_up_detection_accuracy(test_queries: List[Dict]) -> Dict:
    """Evaluate accuracy of the follow-up query detection system."""
    correct = 0
    total = 0

    for q in test_queries:
        expected = q.get("expected_behavior", "")
        is_expected_followup = expected == "follow_up"
        detected_followup = is_follow_up_query(q["question"])

        total += 1
        if is_expected_followup == detected_followup:
            correct += 1

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "total_queries": total,
    }


def document_type_filtering_accuracy(
    test_queries: List[Dict],
    retriever: ContextExpandingHybridRetriever,
) -> Dict:
    """Evaluate document type filtering (email vs manual).

    Tests queries that should filter by doc_type and verifies
    the retrieved chunks match the expected type.
    """
    correct_filters = 0
    total_filters = 0
    type_precision = {"email": [], "manual": []}

    for q in test_queries:
        expected_type = q.get("expected_doc_type")
        if expected_type not in ("email", "manual"):
            continue

        query = q["question"]
        docs = retriever.get_relevant_documents_by_type(query, expected_type)

        total_filters += 1
        if docs:
            matching = sum(1 for d in docs if d.metadata.get("doc_type") == expected_type)
            precision = matching / len(docs)
            type_precision[expected_type].append(precision)
            if precision > 0.5:
                correct_filters += 1

    return {
        "filter_accuracy": correct_filters / total_filters if total_filters > 0 else 0.0,
        "email_precision": float(np.mean(type_precision["email"])) if type_precision["email"] else 0.0,
        "manual_precision": float(np.mean(type_precision["manual"])) if type_precision["manual"] else 0.0,
        "total_queries": total_filters,
    }


def compute_all_custom_metrics(
    test_queries: List[Dict],
    retriever: ContextExpandingHybridRetriever,
) -> Dict:
    """Run all custom metrics and return aggregated results."""
    # Filter queries by category for appropriate tests
    edge_cases = [q for q in test_queries if q.get("category") == "edge_case"]
    non_edge = [q for q in test_queries if q.get("category") != "edge_case"]
    with_expected_db = [q for q in non_edge if q.get("expected_db")]
    with_expected_type = [q for q in non_edge if q.get("expected_doc_type")]

    results = {}

    if with_expected_db:
        results["dual_db_routing"] = dual_db_routing_accuracy(with_expected_db, retriever)

    if non_edge:
        results["context_expansion"] = context_expansion_impact(non_edge[:20], retriever)

    if edge_cases:
        results["vague_detection"] = vague_detection_accuracy(edge_cases)
        results["follow_up_detection"] = follow_up_detection_accuracy(edge_cases)

    if with_expected_type:
        results["doc_type_filtering"] = document_type_filtering_accuracy(with_expected_type, retriever)

    return results
