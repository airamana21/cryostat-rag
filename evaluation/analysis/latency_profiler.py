"""
Latency profiler for measuring RAG pipeline component timing.

Instruments each stage of the query pipeline and computes
percentile latency statistics.
"""
import os
import sys
import time
from typing import List, Dict, Callable, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import evaluation.env_setup  # noqa: F401 — must precede rag imports

import rag
from rag import (
    ContextExpandingHybridRetriever,
    ConversationHistory,
    VertexAIGeminiLLM,
    is_follow_up_query,
    extract_images_from_chunks,
    EXPAND_CONTEXT_BEFORE,
    EXPAND_CONTEXT_AFTER,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from evaluation.eval_config import LATENCY_PERCENTILES, LATENCY_WARMUP_QUERIES


PROFILED_STAGES = [
    "followup_detection",
    "doc_type_detection",
    "retrieval",
    "context_expansion",
    "prompt_assembly",
    "llm_generation",
    "image_extraction",
    "total",
]


def profile_single_query(
    query: str,
    retriever: ContextExpandingHybridRetriever,
    llm: VertexAIGeminiLLM,
    conversation_history: ConversationHistory = None,
) -> Dict[str, float]:
    """Profile a single query, returning timing for each stage in seconds."""
    if conversation_history is None:
        conversation_history = ConversationHistory()

    timings = {}
    total_start = time.perf_counter()

    # 1. Follow-up detection
    t0 = time.perf_counter()
    is_followup = is_follow_up_query(query)
    effective_query = query
    timings["followup_detection"] = time.perf_counter() - t0

    # 2. Document type detection
    t0 = time.perf_counter()
    lower_q = query.lower()
    email_keywords = ["email", "emails", "uhd", "correspondence", "message", "communication"]
    manual_keywords = ["manual", "manuals", "documentation", "guide", "handbook", "instruction"]
    email_score = sum(1 for k in email_keywords if k in lower_q)
    manual_score = sum(1 for k in manual_keywords if k in lower_q)
    doc_type_filter = None
    if email_score > 0 and email_score > manual_score:
        doc_type_filter = "email"
    elif manual_score > 0 and manual_score > email_score:
        doc_type_filter = "manual"
    timings["doc_type_detection"] = time.perf_counter() - t0

    # 3. Retrieval (vector search)
    t0 = time.perf_counter()
    procedural_keywords = ["how", "procedure", "steps", "process", "install", "setup"]
    is_procedural = any(k in lower_q for k in procedural_keywords)
    if is_procedural:
        raw_docs = retriever.coarse_retriever.get_relevant_documents(effective_query)
        db = retriever.coarse_db
    else:
        raw_docs = retriever.fine_retriever.get_relevant_documents(effective_query)
        db = retriever.fine_db
    timings["retrieval"] = time.perf_counter() - t0

    # 4. Context expansion
    t0 = time.perf_counter()
    before = EXPAND_CONTEXT_BEFORE
    after = EXPAND_CONTEXT_AFTER + (1 if is_procedural else 0)
    docs = retriever.expand_context(raw_docs, db, before=before, after=after)
    timings["context_expansion"] = time.perf_counter() - t0

    # 5. Prompt assembly
    t0 = time.perf_counter()
    context_text = "\n\n".join(
        f"[{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}] {d.page_content}"
        for d in docs
    )
    prompt = f"""You are a helpful assistant for answering questions about cryogenic equipment.

Context:
{context_text}

Question: {effective_query}

Answer:"""
    timings["prompt_assembly"] = time.perf_counter() - t0

    # 6. LLM generation
    t0 = time.perf_counter()
    try:
        answer = llm._call(prompt)
    except Exception:
        answer = "Error"
    timings["llm_generation"] = time.perf_counter() - t0

    # 7. Image extraction
    t0 = time.perf_counter()
    images = extract_images_from_chunks(docs)
    timings["image_extraction"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - total_start

    return timings


def profile_queries(
    queries: List[str],
    retriever: ContextExpandingHybridRetriever,
    llm: VertexAIGeminiLLM,
    warmup: int = LATENCY_WARMUP_QUERIES,
    status_callback: Callable = None,
) -> Dict:
    """Profile multiple queries and compute statistics.

    Args:
        queries: List of query strings.
        retriever: The RAG retriever.
        llm: The LLM.
        warmup: Number of warmup queries to discard.
        status_callback: Optional callback(current, total) for progress.

    Returns:
        Dict with per-stage percentile statistics and raw timings.
    """
    all_timings = {stage: [] for stage in PROFILED_STAGES}

    for i, query in enumerate(queries):
        if status_callback:
            status_callback(i + 1, len(queries))

        timings = profile_single_query(query, retriever, llm)

        # Skip warmup queries
        if i >= warmup:
            for stage in PROFILED_STAGES:
                if stage in timings:
                    all_timings[stage].append(timings[stage])

    # Compute statistics
    results = {}
    for stage in PROFILED_STAGES:
        values = all_timings[stage]
        if not values:
            results[stage] = {"mean": 0.0, "std": 0.0, "percentiles": {}}
            continue

        arr = np.array(values)
        percentiles = {}
        for p in LATENCY_PERCENTILES:
            percentiles[f"p{p}"] = float(np.percentile(arr, p))

        results[stage] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "percentiles": percentiles,
            "n_samples": len(arr),
        }

    results["_raw_timings"] = {stage: values for stage, values in all_timings.items()}

    return results


def format_latency_table(results: Dict) -> str:
    """Format latency results as a markdown table."""
    lines = [
        "| Stage | Mean (s) | Std (s) | P50 (s) | P95 (s) | P99 (s) |",
        "|-------|----------|---------|---------|---------|---------|",
    ]

    for stage in PROFILED_STAGES:
        data = results.get(stage, {})
        mean = data.get("mean", 0.0)
        std = data.get("std", 0.0)
        p = data.get("percentiles", {})
        p50 = p.get("p50", 0.0)
        p95 = p.get("p95", 0.0)
        p99 = p.get("p99", 0.0)
        lines.append(f"| {stage} | {mean:.4f} | {std:.4f} | {p50:.4f} | {p95:.4f} | {p99:.4f} |")

    return "\n".join(lines)
