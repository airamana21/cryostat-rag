"""
Generation quality metrics for evaluating RAG responses.

Implements RAGAS-style faithfulness, answer relevancy,
BERTScore, and semantic similarity using Vertex AI embeddings.
"""
import os
import sys
import json
from typing import List, Dict, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import evaluation.env_setup  # noqa: F401 — must precede rag imports

from rag import VertexAIGeminiLLM, VertexAIEmbeddings


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ─── RAGAS Faithfulness ─────────────────────────────────────────────────────

CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the following answer. Each claim should be a single, atomic, verifiable statement.

Answer: {answer}

List each claim on its own line, numbered. Output ONLY the numbered claims, nothing else.
Example:
1. The compressor model is XYZ-500.
2. The operating temperature is -196°C.
"""

CLAIM_VERIFICATION_PROMPT = """Given the following context from source documents, determine if the claim is SUPPORTED or NOT SUPPORTED by the context.

Context:
{context}

Claim: {claim}

Respond with exactly one word: SUPPORTED or NOT_SUPPORTED"""


def ragas_faithfulness(
    answer: str,
    context_docs: List,
    llm: VertexAIGeminiLLM,
) -> float:
    """RAGAS faithfulness: fraction of claims in the answer supported by context.

    1. Extract atomic claims from the answer
    2. Verify each claim against the retrieved context
    3. Score = num_supported / total_claims
    """
    if not answer.strip() or not context_docs:
        return 0.0

    # Step 1: Extract claims
    extraction_prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
    try:
        claims_text = llm._call(extraction_prompt)
    except Exception:
        return 0.0

    claims = []
    for line in claims_text.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            # Strip numbering
            claim = line.lstrip("0123456789.): ").strip()
            if claim:
                claims.append(claim)

    if not claims:
        return 1.0  # No claims to verify — trivially faithful

    # Step 2: Build context string
    context_text = "\n\n".join(doc.page_content for doc in context_docs)

    # Step 3: Verify each claim
    supported = 0
    for claim in claims:
        prompt = CLAIM_VERIFICATION_PROMPT.format(context=context_text, claim=claim)
        try:
            verdict = llm._call(prompt).strip().upper()
            if "SUPPORTED" in verdict and "NOT" not in verdict:
                supported += 1
        except Exception:
            pass

    return supported / len(claims)


# ─── RAGAS Answer Relevancy ─────────────────────────────────────────────────

QUESTION_GENERATION_PROMPT = """Given the following answer, generate {n} questions that this answer could be responding to. The questions should be diverse but all answerable by this text.

Answer: {answer}

Generate exactly {n} questions, one per line, numbered. Output ONLY the questions."""


def ragas_answer_relevancy(
    query: str,
    answer: str,
    embeddings: VertexAIEmbeddings,
    llm: VertexAIGeminiLLM,
    n_questions: int = 3,
) -> float:
    """RAGAS answer relevancy: embedding similarity between original query
    and hypothetical questions generated from the answer.

    High similarity means the answer is relevant to the question asked.
    """
    if not answer.strip():
        return 0.0

    prompt = QUESTION_GENERATION_PROMPT.format(answer=answer, n=n_questions)
    try:
        generated_text = llm._call(prompt)
    except Exception:
        return 0.0

    generated_questions = []
    for line in generated_text.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            q = line.lstrip("0123456789.): ").strip()
            if q:
                generated_questions.append(q)

    if not generated_questions:
        return 0.0

    # Embed original query and generated questions
    try:
        query_embedding = embeddings.embed_query(query)
        question_embeddings = embeddings.embed_documents(generated_questions)
    except Exception:
        return 0.0

    # Average cosine similarity
    similarities = [_cosine_similarity(query_embedding, qe) for qe in question_embeddings]
    return float(np.mean(similarities)) if similarities else 0.0


# ─── Semantic Similarity ────────────────────────────────────────────────────

def semantic_similarity(
    generated_answer: str,
    reference_answer: str,
    embeddings: VertexAIEmbeddings,
) -> float:
    """Cosine similarity between embeddings of generated and reference answers."""
    if not generated_answer.strip() or not reference_answer.strip():
        return 0.0
    try:
        gen_emb = embeddings.embed_query(generated_answer)
        ref_emb = embeddings.embed_query(reference_answer)
        return _cosine_similarity(gen_emb, ref_emb)
    except Exception:
        return 0.0


# ─── BERTScore (optional, requires bert-score package) ──────────────────────

def bertscore_f1(
    generated_answer: str,
    reference_answer: str,
    model_type: str = "roberta-large",
) -> float:
    """Compute BERTScore F1 between generated and reference answers.

    Uses roberta-large (standard BERTScore model) — much faster than deberta-xlarge.
    Requires: pip install bert-score
    """
    if not generated_answer.strip() or not reference_answer.strip():
        return 0.0
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            [generated_answer], [reference_answer],
            model_type=model_type, verbose=False,
            lang="en",
        )
        return float(F1[0])
    except ImportError:
        print("Warning: bert-score package not installed. Run: pip install bert-score")
        return 0.0
    except Exception as e:
        import traceback
        print(f"Warning: BERTScore failed with {type(e).__name__}: {e}")
        traceback.print_exc()
        return 0.0


# ─── Aggregate ──────────────────────────────────────────────────────────────

def compute_generation_metrics(
    results: List[Dict],
    llm: VertexAIGeminiLLM,
    embeddings: VertexAIEmbeddings,
    use_bertscore: bool = True,
) -> Dict:
    """Compute all generation metrics across evaluation results.

    Args:
        results: List of dicts, each with:
            - "query": original question
            - "answer": generated answer
            - "reference_answer": gold/reference answer (optional)
            - "context_docs": list of retrieved Document objects
        llm: LLM for faithfulness/relevancy computation
        embeddings: Embedding model for similarity
        use_bertscore: Whether to compute BERTScore (slower)

    Returns:
        Dict of metric_name -> average value across queries.
    """
    n = len(results)
    if n == 0:
        return {}

    faith_scores = []
    relevancy_scores = []
    sim_scores = []
    bert_scores = []

    for r in results:
        # Faithfulness
        faith = ragas_faithfulness(r["answer"], r.get("context_docs", []), llm)
        faith_scores.append(faith)

        # Answer relevancy
        rel = ragas_answer_relevancy(r["query"], r["answer"], embeddings, llm)
        relevancy_scores.append(rel)

        # Semantic similarity (if reference available)
        ref = r.get("reference_answer")
        if ref:
            sim = semantic_similarity(r["answer"], ref, embeddings)
            sim_scores.append(sim)

            if use_bertscore:
                bs = bertscore_f1(r["answer"], ref)
                bert_scores.append(bs)

    metrics = {
        "faithfulness": float(np.mean(faith_scores)) if faith_scores else 0.0,
        "answer_relevancy": float(np.mean(relevancy_scores)) if relevancy_scores else 0.0,
    }
    if sim_scores:
        metrics["semantic_similarity"] = float(np.mean(sim_scores))
    if bert_scores:
        metrics["bertscore_f1"] = float(np.mean(bert_scores))

    return metrics


def compute_per_query_generation_metrics(
    query: str,
    answer: str,
    context_docs: List,
    llm: VertexAIGeminiLLM,
    embeddings: VertexAIEmbeddings,
    reference_answer: Optional[str] = None,
    use_bertscore: bool = False,
) -> Dict:
    """Compute generation metrics for a single query."""
    metrics = {
        "faithfulness": ragas_faithfulness(answer, context_docs, llm),
        "answer_relevancy": ragas_answer_relevancy(query, answer, embeddings, llm),
    }
    if reference_answer:
        metrics["semantic_similarity"] = semantic_similarity(answer, reference_answer, embeddings)
        if use_bertscore:
            metrics["bertscore_f1"] = bertscore_f1(answer, reference_answer)
    return metrics
