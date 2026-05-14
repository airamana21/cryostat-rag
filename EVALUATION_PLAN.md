# Cryostat RAG Automated Evaluation Framework

## Context

The Cryostat AI Support Bot uses a custom RAG pipeline with several novel features (dual vector databases, context expansion, vague query detection, procedural routing). To publish an academic paper on this system, we need quantifiable performance metrics proving its value over general-purpose solutions (NotebookLM, ChatGPT with uploaded files). Since there are no active users to collect real usage data, we need an automated evaluation pipeline that generates test data, runs benchmarks, and produces publication-ready metrics.

The evaluation design draws from ChemCrow's methodology (Nature Machine Intelligence), the RAGAS framework, RAGBench's TRACe metrics, and standard IR evaluation practices.

---

## File Structure

```
evaluation/
├── __init__.py
├── env_setup.py                      # Local env var setup (must import before rag)
├── eval_config.py                    # All configuration constants
├── dataset/
│   ├── __init__.py
│   ├── synthetic_generator.py        # Auto-generate Q&A from documents
│   ├── gold_standard.json            # Manually curated Q&A (created after first run)
│   └── gold_standard_template.json   # Template with instructions
├── metrics/
│   ├── __init__.py
│   ├── retrieval_metrics.py          # Precision@K, MRR, NDCG, Hit Rate
│   ├── generation_metrics.py         # Faithfulness, relevancy, BERTScore
│   ├── custom_metrics.py             # Feature-specific metrics (dual-DB routing, context expansion, etc.)
│   └── llm_judge.py                  # LLM-as-judge with rubric scoring
├── baselines/
│   ├── __init__.py
│   └── baseline_variants.py          # Ablation variants (no-expansion, single-DB, vanilla LLM, etc.)
├── analysis/
│   ├── __init__.py
│   ├── statistics.py                 # Bootstrap CI, p-values, effect sizes
│   ├── latency_profiler.py           # Component-level timing
│   └── report_generator.py           # Markdown tables, matplotlib charts
├── runner/
│   ├── __init__.py
│   └── run_evaluation.py             # Main orchestrator script
└── outputs/                          # Generated reports, charts, JSON results
```

---

## Step 1: Configuration (`eval_config.py`)

Define all evaluation parameters in one place:
- Dataset sizes: 200 synthetic Q&A (80 factual, 70 procedural, 30 multi-hop, 20 vague/edge-case)
- Gold standard: 30-50 manually curated Q&A pairs (created iteratively)
- Ablation runs: 50 queries per variant
- Latency: 100 query runs
- Bootstrap iterations: 1000 for confidence intervals
- LLM judge model: Gemini 2.5 Pro (same model, separate call with judge prompt)
- Scoring scale: 1-5 discrete (per LLM-as-judge best practices)

---

## Step 2: Synthetic Test Dataset Generation (`dataset/synthetic_generator.py`)

Generate Q&A pairs automatically from the cryostat documents without manual labeling.

**Approach**: Use the LLM itself (Gemini) to generate questions from document chunks, then validate them.

**Implementation**:
1. Load all documents via existing `load_and_split_pdfs()` from `rag.py`
2. For each document chunk, prompt the LLM to generate questions of specific types:
   - **Factual** (40%): "Given this text, generate a specific factual question whose answer is found verbatim in the text" (targets fine DB)
   - **Procedural** (35%): "Generate a how-to question about the procedure described in this text" (targets coarse DB)
   - **Multi-hop** (15%): Take 2-3 related chunks, ask questions requiring synthesis across them
   - **Edge cases** (10%): Vague queries ("tell me about the cryostat"), unanswerable queries, follow-up sequences
3. For each generated question, also generate a **reference answer** grounded in the source chunk(s)
4. Store the source chunk IDs and metadata alongside each Q&A pair for retrieval evaluation
5. Validate: discard questions where the LLM cannot regenerate a consistent answer from the source chunks

**Output format** (`synthetic_qa.json`):
```json
{
  "id": "syn_001",
  "question": "What is the model number of the compressor?",
  "reference_answer": "The compressor model is...",
  "category": "factual",
  "source_chunks": [{"doc_id": "manual.pdf", "chunk_index": 42, "page": 5}],
  "expected_db": "fine",
  "expected_doc_type": "manual"
}
```

---

## Step 3: Retrieval Metrics (`metrics/retrieval_metrics.py`)

Evaluate how well the retriever finds the right chunks.

**Metrics to implement**:
- **Precision@K**: Fraction of top-K retrieved chunks that are relevant (match source chunks)
- **Recall@K**: Fraction of all relevant chunks that appear in top-K
- **Hit Rate@K**: Binary per-query -- did at least one relevant chunk appear in top-K?
- **MRR (Mean Reciprocal Rank)**: 1/rank of first relevant chunk, averaged across queries
- **NDCG@K**: Weighted rank-aware metric giving more credit to relevant chunks ranked higher

**Relevance determination**: A retrieved chunk is "relevant" if it shares the same `doc_id` and its `chunk_index` is within +-2 of a source chunk's index (accounts for context expansion naturally pulling adjacent chunks).

---

## Step 4: Generation Metrics (`metrics/generation_metrics.py`)

Evaluate the quality of the LLM's generated answers.

**Metrics**:
- **RAGAS Faithfulness**: Break the generated answer into claims, check each claim is supported by the retrieved context. Score = fraction of supported claims.
- **RAGAS Answer Relevancy**: Generate N hypothetical questions from the answer, compute embedding similarity between those questions and the original query.
- **BERTScore**: Compute precision/recall/F1 of token-level embeddings between generated answer and reference answer.
- **Semantic Similarity**: Cosine similarity between embeddings of generated answer and reference answer (using Vertex AI text-embedding-005).

---

## Step 5: LLM-as-Judge (`metrics/llm_judge.py`)

Use Gemini as an automated evaluator with a structured rubric.

**Scoring rubric** (1-5 discrete scale per dimension):
1. **Factual Accuracy**: Are the facts in the answer correct based on the source documents?
2. **Completeness**: Does the answer fully address the question?
3. **Clarity**: Is the answer well-organized and easy to understand?
4. **Source Grounding**: Does the answer stick to information from the documents (no hallucination)?

**Implementation**:
- Prompt the judge LLM with: question, generated answer, retrieved context, reference answer (if available)
- Require chain-of-thought reasoning before each score
- Compute inter-run consistency by scoring the same responses 3 times and measuring agreement
- **Calibration**: Score the gold standard answers first; compute Spearman correlation between judge scores and human ratings to report judge reliability

**Honest limitations** (per ChemCrow findings): Report the judge's calibration metrics alongside results. LLM judges tend to be more lenient and may not catch subtle domain errors. Supplement with domain expert spot-checks for final publication.

---

## Step 6: Feature Ablation Study (`baselines/baseline_variants.py`)

Test each novel feature by disabling it and measuring the impact. This is the most publication-worthy section -- it proves each feature contributes measurably.

**Ablation variants** (each wraps the existing retriever/pipeline with a modification):

| Variant | What's Changed | How |
|---------|---------------|-----|
| **Full System** | Baseline -- all features enabled | Use `process_query()` as-is |
| **Single-DB (Fine Only)** | Disable dual-DB routing; always use fine DB | Override `get_relevant_documents()` to skip procedural keyword check |
| **Single-DB (Coarse Only)** | Always use coarse DB | Override to always use coarse retriever |
| **No Context Expansion** | Set EXPAND_CONTEXT_BEFORE=0, EXPAND_CONTEXT_AFTER=0 | Temporarily patch module-level variables |
| **No Follow-up Reformulation** | Skip query reformulation for follow-ups | Bypass the reformulation prompt in `process_query()` |
| **Vanilla LLM (No RAG)** | Send query directly to Gemini with no retrieved context | Call `VertexAIGeminiLLM._call()` directly with just the question |

**For each variant**: Run the same set of test queries, compute all metrics, compare against full system using statistical tests.

---

## Step 7: Latency Profiling (`analysis/latency_profiler.py`)

Instrument `process_query()` to measure time spent in each stage.

**Components to time**:
1. Follow-up detection + query reformulation
2. Document type detection
3. Vector similarity search (retrieval)
4. Context expansion (neighbor chunk fetching)
5. LLM prompt assembly
6. LLM generation (API call)
7. Image extraction
8. **Total end-to-end**

**Implementation**: Wrap each stage with `time.perf_counter()` calls. Run 100 queries, compute P50, P95, P99, mean, and std dev for each component.

---

## Step 8: Statistical Analysis (`analysis/statistics.py`)

Ensure all reported metrics have proper statistical rigor.

- **Bootstrap confidence intervals** (95%): Resample metric values 1000 times, report 2.5th and 97.5th percentiles
- **Paired t-test / Wilcoxon signed-rank test**: Compare full system vs. each ablation variant on the same queries
- **Effect size** (Cohen's d): Quantify practical significance alongside p-values
- **Inter-judge agreement**: If running LLM judge multiple times, compute Krippendorff's alpha or weighted Cohen's kappa

---

## Step 9: Report Generation (`analysis/report_generator.py`)

Produce publication-ready outputs.

**Markdown report** with:
- Summary statistics table (all metrics, all variants)
- Ablation study results with delta and CI
- Latency breakdown table
- LLM judge calibration metrics

**Matplotlib charts**:
- Bar chart: Full system vs. each ablation variant per metric (with error bars)
- Radar/spider chart: Multi-dimensional metric comparison
- Latency distribution histogram
- Box plots for metric distributions by query category

**JSON export**: Raw results for further analysis or re-plotting.

---

## Step 10: Main Runner (`runner/run_evaluation.py`)

Single entry point that orchestrates the entire pipeline:

1. Initialize RAG system (load vectorstores + LLM)
2. Generate synthetic dataset (or load cached)
3. Run retrieval evaluation on all queries
4. Run generation evaluation (full system) on all queries
5. Run LLM-as-judge scoring
6. Run ablation study (all variants)
7. Run latency profiling
8. Compute statistics (CIs, p-values, effect sizes)
9. Generate report + charts
10. Export all results to JSON

**CLI interface**: `python -m evaluation.runner.run_evaluation [--skip-generation] [--ablation-only] [--latency-only] [--num-queries N]`

---

## Dependencies

```
# evaluation dependencies (add to requirements.txt)
bert-score>=0.3.13
scipy>=1.8.0
matplotlib>=3.5.0
seaborn>=0.12.0
tqdm>=4.60.0
```

No changes needed to existing `rag.py` or `app.py` -- the evaluation framework imports from them read-only.

---

## Key Design Decisions

1. **Why not RAGAS library directly?** RAGAS requires OpenAI by default; we use Vertex AI/Gemini. Implementing the metrics directly gives us control and avoids dependency conflicts.
2. **Why LLM-as-judge despite ChemCrow's warning?** ChemCrow found GPT-4 failed as a judge for chemistry. We mitigate this by: (a) reporting calibration metrics, (b) using a structured rubric, (c) supplementing with automated metrics (BERTScore, retrieval metrics) that don't rely on LLM judgment.
3. **Why ablation over baseline comparison with NotebookLM?** NotebookLM doesn't have a programmatic API for fair automated comparison. Ablation proves feature value through controlled experiments, which is more rigorous for publication.
4. **Statistical rigor**: Bootstrap CIs + effect sizes + paired tests follow the methodology of published RAG papers (RAGBench, FRAMES).
