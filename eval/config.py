"""Central configuration for the cryostat RAG evaluation suite."""
import os

# ─── GCP ─────────────────────────────────────────────────────────────────────
GCP_PROJECT  = os.getenv("GCP_PROJECT_ID", "mf-crucible")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# ─── Storage ──────────────────────────────────────────────────────────────────
GCS_BUCKET     = os.getenv("GCS_BUCKET_NAME", "attocube-rag-pdfs")
GCS_PDF_PREFIX = "pdfs/"

# Local PDF cache used during dataset generation (within eval/, gitignored)
PDF_CACHE_DIR = os.path.join(os.path.dirname(__file__), "pdf_cache")

# ─── Models ───────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-pro"
EMBED_MODEL  = "text-embedding-005"

# ─── Dataset generation ───────────────────────────────────────────────────────
N_FACTUAL    = 40
N_PROCEDURAL = 30
N_MULTIHOP   = 20
N_EDGE_CASE  = 10
MIN_CHUNK_CHARS = 100       # skip chunks shorter than this
GENERATION_RETRIES = 2      # JSON parse retry attempts per chunk

# ─── Retrieval evaluation ──────────────────────────────────────────────────────
K_VALUES = [1, 3, 5, 10]
NDCG_K = 10  # primary nDCG cutoff reported in the paper
# A retrieved chunk counts as "hit" if its text shares this fraction of
# unigram overlap with the source_chunk_text used to generate the question.
RETRIEVAL_OVERLAP_THRESHOLD = 0.25

# ─── Single-granularity index (naive-RAG baseline and -dual_granularity ablation) ─
# Standard naive-RAG chunking: one mid-size index, no fine/coarse split.
SINGLE_CHUNK_SIZE    = 1500
SINGLE_CHUNK_OVERLAP = 200
SINGLE_INDEX_K       = 8   # top-k for the dense-only naive retriever

# ─── Long-context baseline ────────────────────────────────────────────────────
# How much of the corpus to stuff into the prompt (chars). Gemini 2.5 Pro has a
# very large window; cap to keep cost bounded and avoid pathological prompts.
LONG_CONTEXT_MAX_CHARS = 700_000

# ─── Variant definitions ──────────────────────────────────────────────────────
# Each variant declares:
#   mode      : "rag" | "vanilla" | "long_context"
#   retriever : "hybrid" (production fine+coarse) | "single" (dense-only mid index)
#               | "single_full" (mid index through the full pipeline)
#   toggles   : rag.py module globals to monkeypatch for this variant
#   group     : which result table it feeds — baseline | retrieval | ablation
# Production behaviour = all toggles at their rag.py defaults.
_DEFAULT_TOGGLES = {
    "RETRIEVAL_MODE": "hybrid",
    "USE_QUERY_ROUTING": True,
    "USE_CONTEXT_EXPANSION": True,
    "USE_RERANKING": True,
    "USE_DOMAIN_PROMPT": True,
}

VARIANT_CONFIGS = {
    # ── Table 4: baselines / head-to-head ──
    "full_system": {"mode": "rag", "retriever": "hybrid", "toggles": {}, "group": "baseline"},
    "naive_rag":   {"mode": "rag", "retriever": "single",
                    "toggles": {"USE_DOMAIN_PROMPT": False}, "group": "baseline"},
    "long_context": {"mode": "long_context", "retriever": "none", "toggles": {}, "group": "baseline"},
    "vanilla_llm":  {"mode": "vanilla", "retriever": "none", "toggles": {}, "group": "baseline"},

    # ── Table 3: retrieval-side configurations (routing + expansion off to isolate the retriever) ──
    "dense_only": {"mode": "rag", "retriever": "hybrid", "group": "retrieval",
                   "toggles": {"RETRIEVAL_MODE": "dense", "USE_QUERY_ROUTING": False,
                               "USE_CONTEXT_EXPANSION": False, "USE_RERANKING": False}},
    "bm25_only":  {"mode": "rag", "retriever": "hybrid", "group": "retrieval",
                   "toggles": {"RETRIEVAL_MODE": "bm25", "USE_QUERY_ROUTING": False,
                               "USE_CONTEXT_EXPANSION": False, "USE_RERANKING": False}},
    "rrf_only":   {"mode": "rag", "retriever": "hybrid", "group": "retrieval",
                   "toggles": {"RETRIEVAL_MODE": "hybrid", "USE_QUERY_ROUTING": False,
                               "USE_CONTEXT_EXPANSION": False, "USE_RERANKING": False}},
    "rrf_rerank": {"mode": "rag", "retriever": "hybrid", "group": "retrieval",
                   "toggles": {"RETRIEVAL_MODE": "hybrid", "USE_QUERY_ROUTING": False,
                               "USE_CONTEXT_EXPANSION": False, "USE_RERANKING": True}},

    # ── Table 5: per-component ablations (production minus exactly one component) ──
    "no_bm25":              {"mode": "rag", "retriever": "hybrid", "group": "ablation",
                             "toggles": {"RETRIEVAL_MODE": "dense"}},
    "no_reranking":         {"mode": "rag", "retriever": "hybrid", "group": "ablation",
                             "toggles": {"USE_RERANKING": False}},
    "no_context_expansion": {"mode": "rag", "retriever": "hybrid", "group": "ablation",
                             "toggles": {"USE_CONTEXT_EXPANSION": False}},
    "no_domain_prompt":     {"mode": "rag", "retriever": "hybrid", "group": "ablation",
                             "toggles": {"USE_DOMAIN_PROMPT": False}},
    "no_dual_granularity":  {"mode": "rag", "retriever": "single_full", "group": "ablation",
                             "toggles": {}},
}

# Order in which to run / report variants.
ABLATION_VARIANTS = list(VARIANT_CONFIGS.keys())

# Variants used in --quick smoke tests (fast, exercises every code path once).
QUICK_VARIANTS = ["full_system", "naive_rag", "vanilla_llm", "rrf_rerank", "no_reranking"]
QUICK_N_QUESTIONS = 6      # dataset slice size for --quick
QUICK_BOOTSTRAP_ITERS = 200

# ─── Cost model (per-query operational estimate) ──────────────────────────────
# List prices in USD; VERIFY against current Vertex AI pricing before publication.
# Tokens are estimated as chars/4 when the API does not return usage metadata.
GEMINI_INPUT_USD_PER_1K  = 0.00125
GEMINI_OUTPUT_USD_PER_1K = 0.005
EMBED_USD_PER_1K         = 0.000025
RERANK_USD_PER_QUERY     = 0.001   # Discovery Engine Ranking API, per request
CHARS_PER_TOKEN          = 4.0

# ─── Vertex AI Evaluation SDK ────────────────────────────────────────────────
# Context is embedded in the prompt column so the judge can verify grounding.
# Truncate to keep the judge's input manageable.
MAX_CONTEXT_CHARS = 8000

VERTEX_EXPERIMENT = "cryostat-rag-paper-eval"
# "context_faithfulness" is a custom PointwiseMetric built in score.py;
# the string here is just a label used in export.py display mappings.
VERTEX_METRICS = [
    "context_faithfulness",
    "answer_correctness",
    "question_answering_quality",
    "coherence",
    "fluency",
    "instruction_following",
    "safety",
]

# ─── Statistics ───────────────────────────────────────────────────────────────
BOOTSTRAP_ITERS = 1000
CI_LEVEL        = 0.95

# ─── Output paths ─────────────────────────────────────────────────────────────
EVAL_DIR    = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(EVAL_DIR, "outputs")
PAPER_DIR   = os.path.join(OUTPUTS_DIR, "paper")

DATASET_PATH    = os.path.join(OUTPUTS_DIR, "dataset.json")
RAG_RESULTS_PATH = os.path.join(OUTPUTS_DIR, "rag_results.json")
SCORES_PATH     = os.path.join(OUTPUTS_DIR, "scores.json")
