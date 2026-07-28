"""
RAG pipeline runner.

For each question in the dataset, runs every configured variant (see
eval/config.py VARIANT_CONFIGS) and records:
  - generated_answer
  - context_text (concatenated retrieved chunks actually sent to the LLM)
  - retrieved_sources
  - retrieval metrics (recall@K, precision@K, MRR, nDCG@K) where gold chunks exist

Variants span three groups: baselines (full_system, naive_rag, long_context,
vanilla_llm), retrieval-side configurations (dense_only, bm25_only, rrf_only,
rrf_rerank), and per-component ablations (no_bm25, no_reranking,
no_context_expansion, no_domain_prompt, no_dual_granularity).

Also records operational metrics (per-stage latency, per-query cost) for the
full system, and the dataset-wide clarification rate.

Output: eval/outputs/rag_results.json
"""
import contextlib
import json
import math
import os
import sys
import time
from typing import List, Dict, Optional

# ── env before importing rag ──────────────────────────────────────────────────
os.environ.setdefault("LOCAL_DEV_MODE", "true")
os.environ.setdefault("VECTORSTORE_BASE_DIR",
                      os.path.join(os.path.dirname(__file__), ".rag_store"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag as _rag_module  # keep reference for monkey-patching toggles
from rag import (
    ContextExpandingHybridRetriever,
    ConversationHistory,
    VertexAIGeminiLLM,
    VertexAIEmbeddings,
    process_query,
    needs_clarification,
    verify_or_rebuild_rag,
    load_rag_if_available,
    SEPARATORS,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from vertexai.generative_models import GenerativeModel

from eval.config import (
    K_VALUES, NDCG_K, RETRIEVAL_OVERLAP_THRESHOLD,
    VARIANT_CONFIGS, _DEFAULT_TOGGLES, GEMINI_MODEL,
    RAG_RESULTS_PATH, OUTPUTS_DIR,
    SINGLE_CHUNK_SIZE, SINGLE_CHUNK_OVERLAP, SINGLE_INDEX_K,
    LONG_CONTEXT_MAX_CHARS,
    GEMINI_INPUT_USD_PER_1K, GEMINI_OUTPUT_USD_PER_1K,
    EMBED_USD_PER_1K, RERANK_USD_PER_QUERY, CHARS_PER_TOKEN,
)


# ─── Retrieval scoring helpers ────────────────────────────────────────────────

def _unigrams(text: str) -> set:
    return set(text.lower().split())


def _overlap_ratio(retrieved_text: str, source_text: str) -> float:
    """Fraction of source unigrams present in the retrieved chunk."""
    src = _unigrams(source_text)
    if not src:
        return 0.0
    ret = _unigrams(retrieved_text)
    return len(src & ret) / len(src)


def _is_hit(doc, source_chunk_text: str) -> bool:
    return _overlap_ratio(doc.page_content, source_chunk_text) >= RETRIEVAL_OVERLAP_THRESHOLD


def _gold_chunks(item: Dict) -> List[str]:
    """Return the list of gold chunk texts for an item.

    Multi-hop questions carry two gold chunks; everything else carries one.
    Falls back to the legacy combined `source_chunk_text` field for old datasets.
    """
    golds = item.get("gold_chunks")
    if golds:
        return [g for g in golds if g]
    sct = item.get("source_chunk_text")
    return [sct] if sct else []


def compute_retrieval_metrics(docs: List, gold_chunks: List[str]) -> Dict:
    """Recall@K, Precision@K, MRR, and nDCG@K for a single query.

    Relevance is binary: a retrieved doc is relevant if it overlaps any gold
    chunk above threshold. Recall@K is the fraction of distinct gold chunks
    found within the top-K (equals Hit@K when there is a single gold chunk).

    nDCG uses one-claim-per-gold relevance: each retrieved doc gets at most
    one relevance credit, and each gold chunk can be claimed at most once
    (by the highest-ranked doc that overlaps it). This ensures DCG <= IDCG.
    """
    if not gold_chunks:
        return {}

    n_gold = len(gold_chunks)

    # rel_flags[i] = 1 if docs[i] matches *any* gold (for Recall/Precision/MRR).
    rel_flags = [1 if any(_is_hit(d, g) for g in gold_chunks) else 0 for d in docs]

    # ndcg_rel[i] = 1 if docs[i] claims a gold not yet claimed by a higher-ranked doc.
    # Each gold can be claimed exactly once → DCG <= IDCG.
    unclaimed = list(range(n_gold))
    ndcg_rel = []
    for d in docs:
        claimed = None
        for gi in list(unclaimed):
            if _is_hit(d, gold_chunks[gi]):
                claimed = gi
                break
        if claimed is not None:
            unclaimed.remove(claimed)
            ndcg_rel.append(1)
        else:
            ndcg_rel.append(0)

    metrics = {}

    for k in K_VALUES:
        top_k_docs = docs[:k]
        found = sum(1 for g in gold_chunks
                    if any(_is_hit(d, g) for d in top_k_docs))
        metrics[f"recall@{k}"] = found / n_gold
        rel_k = rel_flags[:k]
        metrics[f"precision@{k}"] = (sum(rel_k) / k) if rel_k else 0.0

    # MRR: reciprocal rank of the first relevant doc.
    mrr = 0.0
    for rank, rel in enumerate(rel_flags, start=1):
        if rel:
            mrr = 1.0 / rank
            break
    metrics["mrr"] = mrr

    # nDCG@k with claim-capped relevance (guaranteed <= 1.0).
    for k in K_VALUES:
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ndcg_rel[:k]))
        ideal_n = min(n_gold, k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
        metrics[f"ndcg@{k}"] = (dcg / idcg) if idcg > 0 else 0.0

    return metrics


# ─── Cost / latency instrumentation ───────────────────────────────────────────

def _estimate_tokens(text: str) -> float:
    return len(text or "") / CHARS_PER_TOKEN


def _query_cost(context: str, question: str, answer: str, used_rerank: bool) -> float:
    """Per-query USD estimate from char-based token counts (see config pricing)."""
    in_tok  = _estimate_tokens(context) + _estimate_tokens(question)
    out_tok = _estimate_tokens(answer)
    emb_tok = _estimate_tokens(question)
    cost = (in_tok / 1000.0) * GEMINI_INPUT_USD_PER_1K
    cost += (out_tok / 1000.0) * GEMINI_OUTPUT_USD_PER_1K
    cost += (emb_tok / 1000.0) * EMBED_USD_PER_1K
    if used_rerank:
        cost += RERANK_USD_PER_QUERY
    return cost


class _StageTimer:
    """Records the first timestamp seen for each pipeline stage via status_callback."""
    def __init__(self):
        self.t0 = time.time()
        self.marks: Dict[str, float] = {}

    def __call__(self, status: str, msg: Optional[str] = None):
        self.marks.setdefault(status, time.time())

    def latencies(self, t_end: float) -> Dict[str, float]:
        rerank_t = self.marks.get("reranking")
        gen_t = self.marks.get("generating")
        total = t_end - self.t0
        if gen_t is None:
            return {"retrieval": total, "rerank": 0.0, "generation": 0.0, "total": total}
        retrieval_end = rerank_t if rerank_t is not None else gen_t
        return {
            "retrieval":  max(0.0, retrieval_end - self.t0),
            "rerank":     max(0.0, (gen_t - rerank_t) if rerank_t is not None else 0.0),
            "generation": max(0.0, t_end - gen_t),
            "total":      total,
        }


# ─── Toggle patching ──────────────────────────────────────────────────────────

@contextlib.contextmanager
def _patched_toggles(toggles: Dict):
    """Set rag module toggles to defaults+overrides for the duration, then restore."""
    full = {**_DEFAULT_TOGGLES, **(toggles or {})}
    saved = {k: getattr(_rag_module, k) for k in full}
    try:
        for k, v in full.items():
            setattr(_rag_module, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(_rag_module, k, v)


# ─── Single-granularity index (naive_rag, no_dual_granularity) ────────────────

_CHROMA_SINGLE_DIR = os.path.join(
    os.environ["VECTORSTORE_BASE_DIR"], "vectorstore", "chroma_single"
)


def _split_single(folder: str) -> List[Document]:
    """Split the corpus into one mid-size granularity with expansion metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SINGLE_CHUNK_SIZE,
        chunk_overlap=SINGLE_CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    docs: List[Document] = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, fname)
        doc_type = "email" if fname.startswith("UHD") else "manual"
        pages = PyPDFLoader(path).load()
        for i, pg in enumerate(pages, start=1):
            pg.metadata.update({"source": fname, "page": i, "doc_type": doc_type})
        chunks = splitter.split_documents(pages)
        for idx, ch in enumerate(chunks):
            ch.metadata["chunk_index"] = idx
            ch.metadata["total_chunks"] = len(chunks)
            ch.metadata["doc_id"] = fname
        docs.extend(chunks)
    return docs


def _build_single_index():
    """Build (or load) the single mid-size Chroma index. Returns (db, docs)."""
    pdf_folder = _rag_module.PDF_CACHE_DIR
    docs = _split_single(pdf_folder)
    emb = VertexAIEmbeddings()
    os.makedirs(_CHROMA_SINGLE_DIR, exist_ok=True)
    if os.path.exists(os.path.join(_CHROMA_SINGLE_DIR, "chroma.sqlite3")):
        db = Chroma(persist_directory=_CHROMA_SINGLE_DIR, embedding_function=emb)
    else:
        print(f"  Building single mid-size index ({len(docs)} chunks)...")
        db = Chroma.from_documents(docs, emb, persist_directory=_CHROMA_SINGLE_DIR)
    return db, docs


class SingleIndexRetriever:
    """Naive-RAG retriever: dense top-k over one index, no rerank, no expansion."""
    def __init__(self, db, k: int = SINGLE_INDEX_K):
        self.fine_db = db
        self.coarse_db = db
        self._r = db.as_retriever(search_kwargs={"k": k})

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self._r.get_relevant_documents(query)

    def get_relevant_documents_with_expansion(self, query, before=None, after=None):
        return self.get_relevant_documents(query)

    def get_relevant_documents_by_type(self, query, doc_type=None):
        return self.get_relevant_documents(query)

    def get_relevant_documents_by_type_with_expansion(self, query, doc_type=None,
                                                      before=None, after=None):
        return self.get_relevant_documents(query)

    def invoke(self, input: str, config=None):
        return self.get_relevant_documents(input)


def _build_single_full_retriever(mid_db, mid_docs) -> ContextExpandingHybridRetriever:
    """Full pipeline over a single index (BM25 + rerank + expansion), one granularity."""
    saved = _rag_module.GLOBAL_BM25_DOCS
    try:
        _rag_module.GLOBAL_BM25_DOCS = mid_docs
        return ContextExpandingHybridRetriever(mid_db, mid_db)
    finally:
        _rag_module.GLOBAL_BM25_DOCS = saved


# ─── Answer generation per variant ────────────────────────────────────────────

def _run_rag(question: str, retriever, llm, with_timing: bool = False) -> Dict:
    """Run process_query and score the docs actually sent to the LLM."""
    history = ConversationHistory()
    timer = _StageTimer() if with_timing else None
    result = process_query(question, retriever, llm, history,
                           debug_mode=False, status_callback=timer)
    t_end = time.time()
    docs = result.get("retrieved_docs", [])
    context_text = "\n\n---\n\n".join(d.page_content for d in docs)
    sources = [
        {"source": d.metadata.get("source", ""), "page": d.metadata.get("page")}
        for d in docs
    ]
    out = {
        "generated_answer": result.get("answer", ""),
        "context_text": context_text,
        "retrieved_sources": sources,
        "retrieved_docs": docs,
    }
    if with_timing and timer is not None:
        used_rerank = bool(getattr(retriever, "rerank", None)) and _rag_module.USE_RERANKING
        out["timing"] = timer.latencies(t_end)
        out["cost_usd"] = _query_cost(context_text, question,
                                      out["generated_answer"], used_rerank)
    return out


def _run_vanilla_llm(question: str) -> Dict:
    """Gemini only — no retrieval, no context."""
    model = GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(
        question,
        generation_config={"temperature": 0.0, "max_output_tokens": 4096},
    )
    return {
        "generated_answer": response.text,
        "context_text": "",
        "retrieved_sources": [],
        "retrieved_docs": [],
    }


def _run_long_context(question: str, corpus_text: str) -> Dict:
    """Gemini with as much of the corpus as fits in the prompt — no retrieval."""
    model = GenerativeModel(GEMINI_MODEL)
    prompt = (
        "You are a helpful assistant. Use the following documents to answer the "
        f"question.\n\nDocuments:\n{corpus_text}\n\nQuestion: {question}"
    )
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.0, "max_output_tokens": 4096},
    )
    return {
        "generated_answer": response.text,
        "context_text": corpus_text[:LONG_CONTEXT_MAX_CHARS],
        "retrieved_sources": [],
        "retrieved_docs": [],
    }


# ─── Variant orchestration ────────────────────────────────────────────────────

def _result_row(item: Dict, run: Dict) -> Dict:
    metrics = compute_retrieval_metrics(run.get("retrieved_docs", []), _gold_chunks(item))
    row = {
        "id":               item["id"],
        "category":         item["category"],
        "question":         item["question"],
        "reference_answer": item.get("reference_answer"),
        "source_doc":       item.get("source_doc"),
        "source_page":      item.get("source_page"),
        "generated_answer": run["generated_answer"],
        "context_text":     run["context_text"],
        "retrieved_sources": run["retrieved_sources"],
        "retrieval":        metrics,
    }
    if "timing" in run:
        row["timing"] = run["timing"]
    if "cost_usd" in run:
        row["cost_usd"] = run["cost_usd"]
    return row


def _run_one(variant: str, cfg: Dict, item: Dict, assets: Dict, with_timing: bool) -> Dict:
    mode = cfg["mode"]
    q = item["question"]
    if mode == "vanilla":
        return _run_vanilla_llm(q)
    if mode == "long_context":
        return _run_long_context(q, assets["corpus_text"])
    # mode == "rag"
    retriever_kind = cfg["retriever"]
    if retriever_kind == "single":
        retriever = assets["single_retriever"]
    elif retriever_kind == "single_full":
        retriever = assets["single_full_retriever"]
    else:
        retriever = assets["hybrid_retriever"]
    with _patched_toggles(cfg.get("toggles", {})):
        return _run_rag(q, retriever, assets["llm"], with_timing=with_timing)


def _clarification_rate(dataset: List[Dict]) -> float:
    """Fraction of dataset questions that the live vagueness check would clarify.

    Computed directly (no LLM) assuming a fresh conversation per question, which
    matches how each eval question is dispatched.
    """
    if not dataset:
        return 0.0
    n = sum(1 for it in dataset if needs_clarification(it["question"]))
    return n / len(dataset)


def run_pipeline(
    dataset: List[Dict],
    variants: Optional[List[str]] = None,
    skip_rebuild: bool = False,
    quick: bool = False,
) -> Dict:
    """Run every configured variant over the dataset and compute retrieval metrics.

    Args:
        dataset: Question dicts from dataset.py.
        variants: Subset of VARIANT_CONFIGS keys to run (default: all).
        skip_rebuild: Reuse the existing local vectorstore instead of rebuilding from GCS.
        quick: Smoke-test mode — caller already sliced the dataset/variants.

    Returns:
        Results dict saved to RAG_RESULTS_PATH.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    variants = variants or list(VARIANT_CONFIGS.keys())

    # Initialize the production hybrid retriever + LLM.
    if skip_rebuild:
        hybrid_retriever, llm, loaded = load_rag_if_available()
        if not loaded:
            print("No local vectorstore found — rebuilding from GCS...")
            hybrid_retriever, llm, _ = verify_or_rebuild_rag()
    else:
        print("Building/verifying vectorstore from GCS...")
        hybrid_retriever, llm, _ = verify_or_rebuild_rag()

    # Build shared assets on demand.
    assets: Dict = {"hybrid_retriever": hybrid_retriever, "llm": llm}

    needs_single = any(VARIANT_CONFIGS[v]["retriever"] in ("single", "single_full")
                       for v in variants)
    if needs_single:
        print("Preparing single-granularity index...")
        mid_db, mid_docs = _build_single_index()
        assets["single_retriever"] = SingleIndexRetriever(mid_db)
        assets["single_full_retriever"] = _build_single_full_retriever(mid_db, mid_docs)

    if any(VARIANT_CONFIGS[v]["mode"] == "long_context" for v in variants):
        corpus_docs = _rag_module.GLOBAL_BM25_DOCS or []
        parts, total = [], 0
        for d in corpus_docs:
            t = d.page_content
            if total + len(t) > LONG_CONTEXT_MAX_CHARS:
                break
            parts.append(t)
            total += len(t)
        assets["corpus_text"] = "\n\n".join(parts)
        print(f"  Long-context corpus: {total} chars from {len(parts)} chunks")

    # Questions with gold chunks (used by every variant for a fair comparison).
    eval_qs = [it for it in dataset if _gold_chunks(it)]
    print(f"\nRunning {len(variants)} variant(s) on {len(eval_qs)} questions "
          f"({len(dataset) - len(eval_qs)} edge-case questions excluded)...")

    variant_results: Dict[str, List[Dict]] = {}
    for variant in variants:
        cfg = VARIANT_CONFIGS[variant]
        with_timing = (variant == "full_system")
        print(f"\n── Variant: {variant} ({cfg['group']}) ──")
        rows = []
        for i, item in enumerate(eval_qs):
            print(f"  [{i+1}/{len(eval_qs)}] {item['question'][:66]}...")
            try:
                run = _run_one(variant, cfg, item, assets, with_timing)
            except Exception as e:
                print(f"    [error] {e}")
                run = {"generated_answer": f"ERROR: {e}", "context_text": "",
                       "retrieved_sources": [], "retrieved_docs": []}
            rows.append(_result_row(item, run))
            time.sleep(0.3)
        variant_results[variant] = rows

    # Operational metrics (from the timed full_system pass).
    ops = {"timing": [], "cost_usd": []}
    for row in variant_results.get("full_system", []):
        if "timing" in row:
            ops["timing"].append(row["timing"])
        if "cost_usd" in row:
            ops["cost_usd"].append(row["cost_usd"])

    output = {
        "variants": variant_results,
        "ops": ops,
        "clarification_rate": _clarification_rate(dataset),
        "meta": {"quick": quick, "n_eval_questions": len(eval_qs),
                 "n_total_questions": len(dataset), "variants": variants},
    }
    with open(RAG_RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: None)

    print(f"\nRAG results saved to {RAG_RESULTS_PATH}")
    return output


def load_rag_results() -> Dict:
    with open(RAG_RESULTS_PATH) as f:
        return json.load(f)
