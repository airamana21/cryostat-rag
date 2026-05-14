"""
Configuration constants for the Cryostat RAG evaluation framework.
"""
import os

# ─── Paths ──────────────────────────────────────────────────────────────────
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(EVAL_DIR)
OUTPUTS_DIR = os.path.join(EVAL_DIR, "outputs")
DATASET_DIR = os.path.join(EVAL_DIR, "dataset")
SYNTHETIC_QA_PATH = os.path.join(DATASET_DIR, "synthetic_qa.json")
GOLD_STANDARD_PATH = os.path.join(DATASET_DIR, "gold_standard.json")

# ─── Dataset Generation ─────────────────────────────────────────────────────
NUM_FACTUAL_QUESTIONS = 80
NUM_PROCEDURAL_QUESTIONS = 70
NUM_MULTIHOP_QUESTIONS = 30
NUM_EDGE_CASE_QUESTIONS = 20
TOTAL_SYNTHETIC_QUESTIONS = (
    NUM_FACTUAL_QUESTIONS
    + NUM_PROCEDURAL_QUESTIONS
    + NUM_MULTIHOP_QUESTIONS
    + NUM_EDGE_CASE_QUESTIONS
)

# Minimum chunk content length to be eligible for question generation
MIN_CHUNK_LENGTH = 50

# Number of validation attempts per generated Q&A pair
VALIDATION_RETRIES = 2

# ─── Evaluation Parameters ──────────────────────────────────────────────────
EVAL_NUM_QUERIES = 100          # Default number of queries per evaluation run
ABLATION_QUERIES_PER_VARIANT = 50
LATENCY_SAMPLE_SIZE = 100

# Retrieval metric parameters
RETRIEVAL_K_VALUES = [1, 3, 5, 10]
CHUNK_INDEX_TOLERANCE = 2       # Retrieved chunk is "relevant" if within +-N of source

# ─── Statistical Analysis ───────────────────────────────────────────────────
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_ITERATIONS = 1000
SIGNIFICANCE_THRESHOLD = 0.05   # p-value threshold

# ─── LLM Judge ──────────────────────────────────────────────────────────────
JUDGE_MODEL_NAME = "gemini-2.5-pro"
JUDGE_TEMPERATURE = 0.0
JUDGE_RUNS_PER_QUERY = 3        # Score each response N times for consistency
JUDGE_SCORE_SCALE = (1, 5)      # Discrete 1-5 scale

JUDGE_DIMENSIONS = [
    "factual_accuracy",
    "completeness",
    "clarity",
    "source_grounding",
]

# ─── Latency Profiling ──────────────────────────────────────────────────────
LATENCY_PERCENTILES = [50, 95, 99]
LATENCY_WARMUP_QUERIES = 3      # Discard first N queries to avoid cold-start bias

# ─── Ablation Variants ──────────────────────────────────────────────────────
ABLATION_VARIANTS = [
    "full_system",
    "fine_db_only",
    "coarse_db_only",
    "no_context_expansion",
    "no_followup_reformulation",
    "vanilla_llm",
]

# ─── Report ─────────────────────────────────────────────────────────────────
CHART_DPI = 150
CHART_STYLE = "seaborn-v0_8-whitegrid"
