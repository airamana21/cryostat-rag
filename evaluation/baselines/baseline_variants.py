"""
Ablation baseline variants for the Cryostat RAG evaluation.

Each variant disables or modifies one feature of the full system to
measure its impact. All variants produce results in the same format
as the full system for direct comparison.
"""
import os
import sys
import time
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import evaluation.env_setup  # noqa: F401 — must precede rag imports

import rag
from rag import (
    ContextExpandingHybridRetriever,
    ConversationHistory,
    VertexAIGeminiLLM,
    VertexAIEmbeddings,
    process_query,
    is_follow_up_query,
    extract_images_from_chunks,
    EXPAND_CONTEXT_BEFORE,
    EXPAND_CONTEXT_AFTER,
)

from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


def run_full_system(
    query: str, retriever, llm, conversation_history: ConversationHistory
) -> Dict:
    """Run the full system (baseline for comparison)."""
    result = process_query(query, retriever, llm, conversation_history, debug_mode=True)
    return result


def run_fine_db_only(
    query: str, retriever: ContextExpandingHybridRetriever, llm, conversation_history: ConversationHistory
) -> Dict:
    """Run with only the fine-grained database (disable dual-DB routing)."""
    # Override: always use fine retriever regardless of procedural keywords
    docs = retriever.fine_retriever.get_relevant_documents(query)
    expanded = retriever.expand_context(
        docs, retriever.fine_db,
        before=rag.EXPAND_CONTEXT_BEFORE,
        after=rag.EXPAND_CONTEXT_AFTER,
    )
    return _generate_response(query, expanded, llm, conversation_history)


def run_coarse_db_only(
    query: str, retriever: ContextExpandingHybridRetriever, llm, conversation_history: ConversationHistory
) -> Dict:
    """Run with only the coarse-grained database."""
    docs = retriever.coarse_retriever.get_relevant_documents(query)
    expanded = retriever.expand_context(
        docs, retriever.coarse_db,
        before=rag.EXPAND_CONTEXT_BEFORE,
        after=rag.EXPAND_CONTEXT_AFTER + 1,
    )
    return _generate_response(query, expanded, llm, conversation_history)


def run_no_context_expansion(
    query: str, retriever: ContextExpandingHybridRetriever, llm, conversation_history: ConversationHistory
) -> Dict:
    """Run with context expansion disabled."""
    saved_before = rag.EXPAND_CONTEXT_BEFORE
    saved_after = rag.EXPAND_CONTEXT_AFTER
    rag.EXPAND_CONTEXT_BEFORE = 0
    rag.EXPAND_CONTEXT_AFTER = 0
    try:
        result = process_query(query, retriever, llm, conversation_history, debug_mode=True)
    finally:
        rag.EXPAND_CONTEXT_BEFORE = saved_before
        rag.EXPAND_CONTEXT_AFTER = saved_after
    return result


def run_no_followup_reformulation(
    query: str, retriever: ContextExpandingHybridRetriever, llm, conversation_history: ConversationHistory
) -> Dict:
    """Run without follow-up query reformulation.

    The query is passed directly to retrieval without being reformulated
    using conversation context, even if it looks like a follow-up.
    """
    docs = retriever.get_relevant_documents(query)
    return _generate_response(query, docs, llm, conversation_history)


def run_vanilla_llm(
    query: str, llm: VertexAIGeminiLLM, conversation_history: ConversationHistory = None
) -> Dict:
    """Run with vanilla LLM (no RAG retrieval at all)."""
    prompt = f"""You are a helpful assistant for answering questions about cryogenic equipment and cryostat systems.

Question: {query}

Provide a helpful and accurate answer based on your general knowledge."""

    try:
        answer = llm._call(prompt)
    except Exception as e:
        answer = f"Error: {e}"

    return {
        "answer": answer,
        "sources": [],
        "images": [],
        "debug_info": {
            "current_chunks": [],
            "previous_chunks": [],
            "doc_type_filter": None,
            "images_found": 0,
        },
    }


def _generate_response(
    query: str, docs: List[Document], llm, conversation_history: ConversationHistory
) -> Dict:
    """Generate a response from retrieved documents (shared helper)."""
    if not docs:
        return {
            "answer": "No relevant information found.",
            "sources": [],
            "images": [],
            "debug_info": {"current_chunks": [], "previous_chunks": [], "doc_type_filter": None, "images_found": 0},
        }

    # Format context
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )

    prompt = f"""You are a helpful assistant for answering questions about cryogenic equipment and cryostat systems.
Use the following context from source documents to answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

    try:
        answer = llm._call(prompt)
    except Exception as e:
        answer = f"Error: {e}"

    # Build sources
    sources = []
    seen = set()
    for doc in docs:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
        if key not in seen:
            sources.append({
                "filename": doc.metadata.get("source", "unknown"),
                "doc_type": doc.metadata.get("doc_type", "manual"),
                "page": doc.metadata.get("page", 0),
            })
            seen.add(key)

    images = extract_images_from_chunks(docs)

    return {
        "answer": answer,
        "sources": sources,
        "images": images,
        "debug_info": {
            "current_chunks": [
                {
                    "chunk_index": d.metadata.get("chunk_index"),
                    "total_chunks": d.metadata.get("total_chunks"),
                    "source": d.metadata.get("source"),
                    "page": d.metadata.get("page"),
                    "doc_type": d.metadata.get("doc_type"),
                    "content_preview": d.page_content[:100],
                }
                for d in docs[:10]
            ],
            "previous_chunks": [],
            "doc_type_filter": None,
            "images_found": len(images),
        },
    }


# ─── Variant Registry ──────────────────────────────────────────────────────

VARIANT_RUNNERS = {
    "full_system": run_full_system,
    "fine_db_only": run_fine_db_only,
    "coarse_db_only": run_coarse_db_only,
    "no_context_expansion": run_no_context_expansion,
    "no_followup_reformulation": run_no_followup_reformulation,
    "vanilla_llm": run_vanilla_llm,
}


def run_variant(
    variant_name: str,
    query: str,
    retriever: ContextExpandingHybridRetriever,
    llm: VertexAIGeminiLLM,
    conversation_history: ConversationHistory = None,
) -> Dict:
    """Run a specific ablation variant by name."""
    if conversation_history is None:
        conversation_history = ConversationHistory()

    runner = VARIANT_RUNNERS.get(variant_name)
    if runner is None:
        raise ValueError(f"Unknown variant: {variant_name}. Available: {list(VARIANT_RUNNERS.keys())}")

    if variant_name == "vanilla_llm":
        return runner(query, llm, conversation_history)
    else:
        return runner(query, retriever, llm, conversation_history)


def run_all_variants(
    query: str,
    retriever: ContextExpandingHybridRetriever,
    llm: VertexAIGeminiLLM,
    variants: List[str] = None,
) -> Dict[str, Dict]:
    """Run all (or specified) ablation variants for a single query.

    Returns a dict mapping variant_name -> result.
    """
    if variants is None:
        variants = list(VARIANT_RUNNERS.keys())

    results = {}
    for variant in variants:
        conversation_history = ConversationHistory()
        results[variant] = run_variant(variant, query, retriever, llm, conversation_history)

    return results
