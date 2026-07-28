"""
Synthetic Q&A dataset generator.

Downloads PDFs from GCS, chunks them via rag.py's splitter, then uses
Gemini to generate (question, reference_answer) pairs per category.

Output: eval/outputs/dataset.json
"""
import json
import os
import random
import sys
import time
from typing import List, Dict, Optional

# ── must set env before importing rag (rag runs vertexai.init at import time) ──
os.environ.setdefault("LOCAL_DEV_MODE", "true")
os.environ.setdefault("VECTORSTORE_BASE_DIR", os.path.join(os.path.dirname(__file__), ".rag_store"))

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import (
    VertexAIGeminiLLM,
    download_pdfs_from_gcs,
    load_and_split_pdfs,
    GCS_BUCKET_NAME,
    GCS_PDF_PREFIX,
)
from eval.config import (
    PDF_CACHE_DIR,
    N_FACTUAL, N_PROCEDURAL, N_MULTIHOP, N_EDGE_CASE,
    MIN_CHUNK_CHARS, GENERATION_RETRIES,
    DATASET_PATH, OUTPUTS_DIR,
)

# ─── Prompts ──────────────────────────────────────────────────────────────────

_FACTUAL_PROMPT = """You are creating evaluation data for a RAG system about cryogenic laboratory equipment (attoDRY cryostats).

Given this excerpt from a technical document, generate ONE specific factual question whose answer is explicitly stated in the text. The question should ask about a specific value, specification, name, or observable fact.

Document excerpt:
{chunk}

Source: {source}, page {page}

Respond with ONLY valid JSON (no markdown, no code fences):
{{"question": "...", "reference_answer": "..."}}"""

_PROCEDURAL_PROMPT = """You are creating evaluation data for a RAG system about cryogenic laboratory equipment (attoDRY cryostats).

Given this excerpt describing a procedure or operation, generate ONE "how-to" question that requires understanding the steps or process described.

Document excerpt:
{chunk}

Source: {source}, page {page}

Respond with ONLY valid JSON (no markdown, no code fences):
{{"question": "...", "reference_answer": "..."}}"""

_MULTIHOP_PROMPT = """You are creating evaluation data for a RAG system about cryogenic laboratory equipment (attoDRY cryostats).

Given TWO excerpts from technical documents, generate ONE question that requires information from BOTH passages to answer fully. The question should connect concepts or specifications across both texts.

Passage A ({source_a}, page {page_a}):
{chunk_a}

Passage B ({source_b}, page {page_b}):
{chunk_b}

Respond with ONLY valid JSON (no markdown, no code fences):
{{"question": "...", "reference_answer": "..."}}"""

# Edge-case questions are template-based (no source chunk needed)
_EDGE_CASE_TEMPLATES = [
    {"question": "what", "category": "edge_case", "behavior": "should_clarify"},
    {"question": "how", "category": "edge_case", "behavior": "should_clarify"},
    {"question": "Tell me about the system", "category": "edge_case", "behavior": "should_clarify"},
    {"question": "What is the current stock price of attocube?", "category": "edge_case", "behavior": "out_of_scope"},
    {"question": "What is the weather like today?", "category": "edge_case", "behavior": "out_of_scope"},
    {"question": "Can you write a poem about cryostats?", "category": "edge_case", "behavior": "out_of_scope"},
    {"question": "huh?", "category": "edge_case", "behavior": "should_clarify"},
    {"question": "What happened?", "category": "edge_case", "behavior": "should_clarify"},
    {"question": "Fix it", "category": "edge_case", "behavior": "should_clarify"},
    {"question": "Is the attoDRY safe?", "category": "edge_case", "behavior": "ambiguous"},
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _call_gemini(llm: VertexAIGeminiLLM, prompt: str) -> Optional[Dict]:
    """Call Gemini and parse JSON response; returns None on failure."""
    for attempt in range(GENERATION_RETRIES):
        try:
            raw = llm._call(prompt)
            # Strip markdown fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except (json.JSONDecodeError, Exception) as e:
            if attempt == GENERATION_RETRIES - 1:
                print(f"    [warn] JSON parse failed after {GENERATION_RETRIES} attempts: {e}")
            time.sleep(1)
    return None


def _source_name(doc) -> str:
    return os.path.basename(doc.metadata.get("source", "unknown"))


def _eligible_chunks(docs, seen_ids: set) -> List:
    """Return chunks with enough content that haven't been used yet."""
    return [
        d for d in docs
        if len(d.page_content) >= MIN_CHUNK_CHARS
        and id(d) not in seen_ids
    ]


# ─── Category generators ──────────────────────────────────────────────────────

def _generate_factual(llm, docs, n: int, seen_ids: set) -> List[Dict]:
    results = []
    candidates = _eligible_chunks(docs, seen_ids)
    random.shuffle(candidates)
    q_id = 1
    for doc in candidates:
        if len(results) >= n:
            break
        prompt = _FACTUAL_PROMPT.format(
            chunk=doc.page_content,
            source=_source_name(doc),
            page=doc.metadata.get("page", "?"),
        )
        parsed = _call_gemini(llm, prompt)
        if parsed and parsed.get("question") and parsed.get("reference_answer"):
            results.append({
                "id": f"f_{q_id:03d}",
                "category": "factual",
                "question": parsed["question"],
                "reference_answer": parsed["reference_answer"],
                "source_doc": _source_name(doc),
                "source_page": doc.metadata.get("page"),
                "source_chunk_text": doc.page_content,
                "gold_chunks": [doc.page_content],
            })
            seen_ids.add(id(doc))
            q_id += 1
            print(f"  factual {len(results)}/{n}: {parsed['question'][:60]}...")
    return results


def _generate_procedural(llm, docs, n: int, seen_ids: set) -> List[Dict]:
    # Prefer coarse chunks for procedural (more context for steps)
    results = []
    candidates = _eligible_chunks(docs, seen_ids)
    # Heuristic: prefer chunks with procedural keywords
    procedural_kw = {"step", "procedure", "install", "connect", "operate",
                     "press", "turn", "open", "close", "ensure", "check", "warning"}
    scored = sorted(
        candidates,
        key=lambda d: sum(1 for w in procedural_kw if w in d.page_content.lower()),
        reverse=True,
    )
    q_id = 1
    for doc in scored:
        if len(results) >= n:
            break
        prompt = _PROCEDURAL_PROMPT.format(
            chunk=doc.page_content,
            source=_source_name(doc),
            page=doc.metadata.get("page", "?"),
        )
        parsed = _call_gemini(llm, prompt)
        if parsed and parsed.get("question") and parsed.get("reference_answer"):
            results.append({
                "id": f"p_{q_id:03d}",
                "category": "procedural",
                "question": parsed["question"],
                "reference_answer": parsed["reference_answer"],
                "source_doc": _source_name(doc),
                "source_page": doc.metadata.get("page"),
                "source_chunk_text": doc.page_content,
                "gold_chunks": [doc.page_content],
            })
            seen_ids.add(id(doc))
            q_id += 1
            print(f"  procedural {len(results)}/{n}: {parsed['question'][:60]}...")
    return results


def _generate_multihop(llm, docs, n: int, seen_ids: set) -> List[Dict]:
    results = []
    candidates = _eligible_chunks(docs, seen_ids)
    # Build pairs from different source documents
    by_doc: Dict[str, List] = {}
    for d in candidates:
        key = _source_name(d)
        by_doc.setdefault(key, []).append(d)

    doc_names = list(by_doc.keys())
    if len(doc_names) < 2:
        print("  [warn] fewer than 2 source docs — skipping multi-hop generation")
        return results

    pairs = []
    for i, name_a in enumerate(doc_names):
        for name_b in doc_names[i + 1:]:
            for da in by_doc[name_a][:3]:
                for db in by_doc[name_b][:3]:
                    pairs.append((da, db))

    random.shuffle(pairs)
    q_id = 1
    for doc_a, doc_b in pairs:
        if len(results) >= n:
            break
        prompt = _MULTIHOP_PROMPT.format(
            chunk_a=doc_a.page_content,
            source_a=_source_name(doc_a),
            page_a=doc_a.metadata.get("page", "?"),
            chunk_b=doc_b.page_content,
            source_b=_source_name(doc_b),
            page_b=doc_b.metadata.get("page", "?"),
        )
        parsed = _call_gemini(llm, prompt)
        if parsed and parsed.get("question") and parsed.get("reference_answer"):
            results.append({
                "id": f"m_{q_id:03d}",
                "category": "multihop",
                "question": parsed["question"],
                "reference_answer": parsed["reference_answer"],
                "source_doc": f"{_source_name(doc_a)} + {_source_name(doc_b)}",
                "source_page": None,
                "source_chunk_text": doc_a.page_content + "\n\n" + doc_b.page_content,
                "gold_chunks": [doc_a.page_content, doc_b.page_content],
                "gold_pages": [
                    {"source": _source_name(doc_a), "page": doc_a.metadata.get("page")},
                    {"source": _source_name(doc_b), "page": doc_b.metadata.get("page")},
                ],
            })
            seen_ids.update({id(doc_a), id(doc_b)})
            q_id += 1
            print(f"  multihop {len(results)}/{n}: {parsed['question'][:60]}...")
    return results


def _generate_edge_cases(n: int) -> List[Dict]:
    templates = (_EDGE_CASE_TEMPLATES * ((n // len(_EDGE_CASE_TEMPLATES)) + 1))[:n]
    return [
        {
            "id": f"e_{i+1:03d}",
            "category": "edge_case",
            "question": t["question"],
            "reference_answer": None,
            "source_doc": None,
            "source_page": None,
            "source_chunk_text": None,
        }
        for i, t in enumerate(templates)
    ]


# ─── Main entry point ─────────────────────────────────────────────────────────

def generate_dataset(pdf_folder: Optional[str] = None) -> List[Dict]:
    """Generate synthetic Q&A dataset from GCS PDFs.

    Args:
        pdf_folder: Override — use local PDFs instead of downloading from GCS.

    Returns:
        List of question dicts saved to DATASET_PATH.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(PDF_CACHE_DIR, exist_ok=True)

    # 1. Get PDFs
    if pdf_folder:
        folder = pdf_folder
        print(f"Using local PDF folder: {folder}")
    else:
        print(f"Downloading PDFs from gs://{GCS_BUCKET_NAME}/{GCS_PDF_PREFIX} ...")
        download_pdfs_from_gcs(GCS_BUCKET_NAME, GCS_PDF_PREFIX, PDF_CACHE_DIR)
        folder = PDF_CACHE_DIR

    # 2. Chunk documents (fine chunks for factual/procedural, coarse for multi-hop pairs)
    print("Chunking documents...")
    fine_docs, coarse_docs = load_and_split_pdfs(folder)
    print(f"  {len(fine_docs)} fine chunks, {len(coarse_docs)} coarse chunks")

    if not fine_docs:
        raise RuntimeError(f"No documents loaded from {folder}. Check the path and GCS access.")

    # 3. Initialize LLM
    llm = VertexAIGeminiLLM()
    seen_ids: set = set()

    # 4. Generate per category
    print(f"\nGenerating {N_FACTUAL} factual questions...")
    factual = _generate_factual(llm, fine_docs, N_FACTUAL, seen_ids)

    print(f"\nGenerating {N_PROCEDURAL} procedural questions...")
    procedural = _generate_procedural(llm, coarse_docs, N_PROCEDURAL, seen_ids)

    print(f"\nGenerating {N_MULTIHOP} multi-hop questions...")
    multihop = _generate_multihop(llm, fine_docs, N_MULTIHOP, seen_ids)

    print(f"\nGenerating {N_EDGE_CASE} edge-case questions...")
    edge_cases = _generate_edge_cases(N_EDGE_CASE)

    dataset = factual + procedural + multihop + edge_cases
    random.shuffle(dataset)

    # Re-assign sequential IDs after shuffle
    for i, item in enumerate(dataset):
        item["id"] = f"q_{i+1:03d}"

    # 5. Save
    with open(DATASET_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    total = len(dataset)
    print(f"\nDataset saved to {DATASET_PATH}")
    print(f"  Total: {total} questions")
    print(f"  Factual: {len(factual)}, Procedural: {len(procedural)}, "
          f"Multi-hop: {len(multihop)}, Edge-case: {len(edge_cases)}")
    return dataset


def load_dataset() -> List[Dict]:
    """Load existing dataset from disk."""
    with open(DATASET_PATH) as f:
        return json.load(f)
