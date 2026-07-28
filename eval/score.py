"""
Scoring module: Vertex AI Evaluation SDK + retrieval/operational aggregation.

Reads eval/outputs/rag_results.json (the {"variants", "ops", "clarification_rate"}
structure produced by pipeline.py), scores generation quality with
vertexai.evaluation.EvalTask, aggregates retrieval metrics (recall@k, precision@k,
MRR, nDCG@k), computes ROUGE-L and citation accuracy locally, aggregates
operational metrics, bootstraps CIs, and runs paired Wilcoxon tests of each
ablation variant against the full system.

Output: eval/outputs/scores.json
"""
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np

os.environ.setdefault("LOCAL_DEV_MODE", "true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vertexai
import pandas as pd
from vertexai.evaluation import EvalTask, PointwiseMetric, PointwiseMetricPromptTemplate

from eval.config import (
    GCP_PROJECT, GCP_LOCATION,
    VERTEX_EXPERIMENT, VERTEX_METRICS, MAX_CONTEXT_CHARS,
    K_VALUES, NDCG_K, BOOTSTRAP_ITERS, QUICK_BOOTSTRAP_ITERS, CI_LEVEL,
    VARIANT_CONFIGS,
    SCORES_PATH, OUTPUTS_DIR,
)

# Bootstrap iteration count; score() lowers this in --quick mode.
_BOOT_ITERS = BOOTSTRAP_ITERS


# ─── Custom Vertex metrics ────────────────────────────────────────────────────
# The built-in "groundedness" metric is binary (0/1) and penalises any
# paraphrase. context_faithfulness awards a continuous 1-5 score so ablation
# comparisons are nuanced. answer_correctness scores the response against the
# authored gold reference — it counters the QA-quality metric's tendency to
# reward fluent but ungrounded bare-LLM answers.

_FAITHFULNESS_CRITERIA = (
    "Context Faithfulness: Every factual claim, number, or procedure step in the "
    "response must be directly supported by the context provided in the user prompt. "
    "Reasonable paraphrase is acceptable; introducing new facts is not."
)

_FAITHFULNESS_RUBRIC = {
    "5": "All information directly supported by context. No external facts added.",
    "4": "Nearly all from context; at most one minor interpretive paraphrase.",
    "3": "Most claims from context; notable synthesis or one unsupported claim.",
    "2": "Some claims match context; significant unsupported content present.",
    "1": "Response largely ignores, contradicts, or extends beyond the context.",
}

CONTEXT_FAITHFULNESS_METRIC = PointwiseMetric(
    metric="context_faithfulness",
    metric_prompt_template=PointwiseMetricPromptTemplate(
        criteria={"context_faithfulness": _FAITHFULNESS_CRITERIA},
        rating_rubric=_FAITHFULNESS_RUBRIC,
        input_variables=["prompt", "response"],
    ),
)

_CORRECTNESS_CRITERIA = (
    "Answer Correctness: Judge whether the response is factually correct and "
    "complete relative to the reference answer. The response need not match the "
    "wording of the reference, but it must convey the same facts, values, and "
    "conclusions. Penalise missing key facts, wrong values, and contradictions."
)

_CORRECTNESS_RUBRIC = {
    "5": "Fully correct and complete: all key facts of the reference are present and accurate.",
    "4": "Correct with a minor omission or imprecision that does not mislead.",
    "3": "Partially correct: some key facts right, but a notable omission or error.",
    "2": "Mostly incorrect: a few details overlap but the core answer is wrong or missing.",
    "1": "Incorrect or contradicts the reference answer.",
}

ANSWER_CORRECTNESS_METRIC = PointwiseMetric(
    metric="answer_correctness",
    metric_prompt_template=PointwiseMetricPromptTemplate(
        criteria={"answer_correctness": _CORRECTNESS_CRITERIA},
        rating_rubric=_CORRECTNESS_RUBRIC,
        input_variables=["prompt", "response", "reference"],
    ),
)

_CUSTOM_METRICS = {
    "context_faithfulness": CONTEXT_FAITHFULNESS_METRIC,
    "answer_correctness": ANSWER_CORRECTNESS_METRIC,
}


# ─── Bootstrap CI ─────────────────────────────────────────────────────────────

def _bootstrap_ci(values: List[float]) -> Dict:
    """Mean and 95% CI via bootstrap resampling."""
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    arr = np.array(values, dtype=float)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True))
             for _ in range(_BOOT_ITERS)]
    alpha = 1 - CI_LEVEL
    return {
        "mean":    float(np.mean(arr)),
        "ci_low":  float(np.percentile(means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n":       len(arr),
    }


# ─── Local text metrics: ROUGE-L + citation accuracy ──────────────────────────

def _lcs_len(a: List[str], b: List[str]) -> int:
    """Length of the longest common subsequence of two token lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(m):
        cur = [0] * (n + 1)
        ai = a[i]
        for j in range(n):
            cur[j + 1] = prev[j] + 1 if ai == b[j] else max(prev[j + 1], cur[j])
        prev = cur
    return prev[n]


def _rouge_l(prediction: str, reference: str) -> float:
    """Token-level ROUGE-L F1 between a prediction and a reference answer."""
    pred = (prediction or "").lower().split()
    ref = (reference or "").lower().split()
    if not pred or not ref:
        return 0.0
    lcs = _lcs_len(pred, ref)
    if lcs == 0:
        return 0.0
    prec = lcs / len(pred)
    rec = lcs / len(ref)
    return 2 * prec * rec / (prec + rec)


def _doc_tokens(doc_name: str) -> List[str]:
    """Significant tokens of a source filename stem (for loose doc matching)."""
    stem = os.path.splitext(doc_name or "")[0].lower()
    return [t for t in re.split(r"[^a-z0-9]+", stem) if len(t) > 3]


def _citation_match(answer: str, gold_doc: Optional[str],
                    gold_page) -> Optional[float]:
    """1.0 if the answer carries a page citation matching the gold source.

    Returns None when the question has no single gold page (e.g. multi-hop), so
    the item is excluded from the citation-accuracy denominator. Requires both a
    page-number citation matching the gold page AND a mention of the gold
    document, matching the domain prompt's "(Source: <file>, page N)" format.
    """
    if not gold_doc or gold_page is None:
        return None
    ans = (answer or "").lower()
    try:
        page = int(gold_page)
    except (TypeError, ValueError):
        return None
    # Page citation: "page 7", "p. 7", "pg 7", or "(..., 7)" near a source.
    page_hit = bool(
        re.search(rf"\b(?:page|pg|p\.?)\s*0*{page}\b", ans)
        or re.search(rf",\s*0*{page}\s*\)", ans)
    )
    doc_toks = _doc_tokens(gold_doc)
    doc_hit = any(t in ans for t in doc_toks) if doc_toks else False
    return 1.0 if (page_hit and doc_hit) else 0.0


# ─── Retrieval aggregation ────────────────────────────────────────────────────

_RETRIEVAL_KEYS = (
    [f"recall@{k}" for k in K_VALUES]
    + [f"precision@{k}" for k in K_VALUES]
    + [f"ndcg@{k}" for k in K_VALUES]
    + ["mrr"]
)


def _agg_retrieval(results: List[Dict]) -> Dict:
    """Aggregate retrieval metrics overall and per category."""
    def _mean_metrics(items: List[Dict]) -> Dict:
        out = {}
        for key in _RETRIEVAL_KEYS:
            vals = [r["retrieval"][key] for r in items
                    if r.get("retrieval") and key in r["retrieval"]]
            if vals:
                out[key] = _bootstrap_ci(vals)
        return out

    scored = [r for r in results if r.get("retrieval")]
    categories = sorted({r["category"] for r in scored})
    per_category = {cat: _mean_metrics([r for r in scored if r["category"] == cat])
                    for cat in categories}
    return {"overall": _mean_metrics(scored), "per_category": per_category}


# ─── Local generation aggregation (ROUGE-L, citation accuracy) ────────────────

def _row_rouge(r: Dict) -> Optional[float]:
    ga = r.get("generated_answer", "")
    ref = r.get("reference_answer")
    if not ga or ga.startswith("ERROR") or not ref:
        return None
    return _rouge_l(ga, ref)


def _row_citation(r: Dict) -> Optional[float]:
    ga = r.get("generated_answer", "")
    if not ga or ga.startswith("ERROR"):
        return None
    return _citation_match(ga, r.get("source_doc"), r.get("source_page"))


def _agg_generation_extra(results: List[Dict]) -> Dict:
    """ROUGE-L vs reference and citation accuracy, overall and per category."""
    def _agg(items: List[Dict]) -> Dict:
        rouge = [v for v in (_row_rouge(r) for r in items) if v is not None]
        cite = [v for v in (_row_citation(r) for r in items) if v is not None]
        out = {}
        if rouge:
            out["rouge_l"] = _bootstrap_ci(rouge)
        if cite:
            out["citation_accuracy"] = _bootstrap_ci(cite)
        return out

    categories = sorted({r["category"] for r in results})
    return {
        "overall": _agg(results),
        "per_category": {cat: _agg([r for r in results if r["category"] == cat])
                         for cat in categories},
    }


# ─── Operational aggregation ──────────────────────────────────────────────────

def _stats(vals: List[float]) -> Dict:
    if not vals:
        return {"median": None, "p95": None, "mean": None, "n": 0}
    a = np.array(vals, dtype=float)
    return {
        "median": float(np.median(a)),
        "p95":    float(np.percentile(a, 95)),
        "mean":   float(np.mean(a)),
        "n":      len(a),
    }


def _agg_ops(ops: Dict, clarification_rate: Optional[float]) -> Dict:
    timing = ops.get("timing", []) if ops else []
    cost = ops.get("cost_usd", []) if ops else []
    stages = ["retrieval", "rerank", "generation", "total"]
    timing_agg = {s: _stats([t[s] for t in timing if isinstance(t, dict) and s in t])
                  for s in stages}
    return {
        "timing": timing_agg,
        "cost_usd": _stats(cost),
        "clarification_rate": clarification_rate,
    }


# ─── Vertex AI generation scoring ─────────────────────────────────────────────

def _run_vertex_eval(results: List[Dict], experiment_suffix: str = "") -> Dict:
    """Run Vertex AI EvalTask over a variant's rows (all carry a reference)."""
    scoreable = [r for r in results
                 if r.get("generated_answer")
                 and not r["generated_answer"].startswith("ERROR")]
    if not scoreable:
        return {"error": "No scoreable results"}

    def _fmt_prompt(question: str, context: str) -> str:
        ctx = context[:MAX_CONTEXT_CHARS] if context else ""
        return f"Context:\n{ctx}\n\nQuestion: {question}" if ctx else question

    df = pd.DataFrame({
        "id":        [r["id"] for r in scoreable],
        "prompt":    [_fmt_prompt(r["question"], r.get("context_text", "")) for r in scoreable],
        "response":  [r["generated_answer"] for r in scoreable],
        "reference": [r.get("reference_answer") or "" for r in scoreable],
    })

    active_metrics = [_CUSTOM_METRICS.get(m, m) for m in VERTEX_METRICS]

    safe_suffix = experiment_suffix.replace("_", "-")
    experiment_name = f"{VERTEX_EXPERIMENT}-{safe_suffix}" if safe_suffix else VERTEX_EXPERIMENT

    print(f"  Vertex EvalTask: {len(scoreable)} items × {len(active_metrics)} metrics...")
    try:
        eval_task = EvalTask(dataset=df, metrics=active_metrics, experiment=experiment_name)
        eval_result = eval_task.evaluate()

        summary = {k: float(v) for k, v in eval_result.summary_metrics.items()
                   if v is not None and not (isinstance(v, float) and np.isnan(v))}

        per_instance_cols = [c for c in eval_result.metrics_table.columns
                             if c.endswith("/score")]
        # Row order is preserved, so attach ids positionally for cross-variant pairing.
        per_instance = eval_result.metrics_table[per_instance_cols].to_dict(orient="records")
        ids = list(df["id"])
        for i, row in enumerate(per_instance):
            row["id"] = ids[i] if i < len(ids) else None

        ci_stats = {}
        for col in per_instance_cols:
            vals = eval_result.metrics_table[col].dropna().tolist()
            ci_stats[col.replace("/score", "")] = _bootstrap_ci(vals)

        return {"summary": summary, "ci": ci_stats, "per_instance": per_instance}

    except Exception as e:
        print(f"  [error] Vertex EvalTask failed: {e}")
        return {"error": str(e)}


# ─── Significance testing ─────────────────────────────────────────────────────

def _nan_to_none(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _wilcoxon_test(a: List[float], b: List[float]) -> Dict:
    """Paired Wilcoxon signed-rank test + Cohen's d on the paired differences."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {"error": "scipy not installed"}

    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    n = min(len(a), len(b))
    if n < 5:
        return {"p_value": None, "cohens_d": None, "note": "too few samples"}
    diff = a[:n] - b[:n]
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": None, "cohens_d": 0.0,
                "note": "identical on this metric"}
    try:
        stat, p = wilcoxon(diff)
    except Exception as e:
        return {"p_value": None, "error": str(e)}
    pooled_std = np.std(diff, ddof=1)
    return {
        "statistic": float(stat),
        "p_value": None if math.isnan(p) else float(p),
        "cohens_d": float(np.mean(diff) / pooled_std) if pooled_std > 0 else 0.0,
        "n": int(n),
    }


def _paired_by_id(full_rows: List[Dict], var_rows: List[Dict], value_fn):
    """Return aligned (full, variant) value lists paired on row id."""
    full_map = {r["id"]: value_fn(r) for r in full_rows}
    var_map = {r["id"]: value_fn(r) for r in var_rows}
    a, b = [], []
    for rid in full_map:
        if rid in var_map and full_map[rid] is not None and var_map[rid] is not None:
            a.append(full_map[rid])
            b.append(var_map[rid])
    return a, b


def _vertex_pairs(full_vertex: Dict, var_vertex: Dict, metric: str):
    """Aligned full vs variant Vertex per-instance scores for a metric, by id."""
    col = f"{metric}/score"
    full_map = {row.get("id"): row.get(col)
                for row in (full_vertex.get("per_instance") or [])}
    var_map = {row.get("id"): row.get(col)
               for row in (var_vertex.get("per_instance") or [])}
    a, b = [], []
    for rid, fv in full_map.items():
        vv = var_map.get(rid)
        if fv is not None and vv is not None:
            a.append(fv)
            b.append(vv)
    return a, b


def _significance_vs_full(full_rows, var_rows, full_vertex, var_vertex) -> Dict:
    """Paired full-vs-variant tests on retrieval, ROUGE-L, citation, and Vertex metrics."""
    tests = {}
    # Retrieval + local generation metrics (paired by id).
    retr_keys = [f"recall@{k}" for k in K_VALUES] + [f"ndcg@{NDCG_K}", "mrr"]
    for key in retr_keys:
        a, b = _paired_by_id(full_rows, var_rows,
                             lambda r, k=key: (r.get("retrieval") or {}).get(k))
        if a and b:
            tests[key] = _wilcoxon_test(a, b)
    for name, fn in (("rouge_l", _row_rouge), ("citation_accuracy", _row_citation)):
        a, b = _paired_by_id(full_rows, var_rows, fn)
        if a and b:
            tests[name] = _wilcoxon_test(a, b)
    # Vertex per-instance metrics (paired by id).
    if full_vertex and var_vertex and "per_instance" in var_vertex:
        for metric in ("context_faithfulness", "answer_correctness",
                       "question_answering_quality"):
            a, b = _vertex_pairs(full_vertex, var_vertex, metric)
            if a and b:
                tests[metric] = _wilcoxon_test(a, b)
    return tests


# ─── Main entry point ─────────────────────────────────────────────────────────

def score(rag_results: Dict, quick: bool = False) -> Dict:
    """Score every variant in rag_results and save to SCORES_PATH.

    Args:
        rag_results: {"variants", "ops", "clarification_rate", "meta"} from pipeline.py.
        quick: smoke-test mode — fewer bootstrap resamples.

    Returns:
        Scores dict saved to SCORES_PATH.
    """
    global _BOOT_ITERS
    _BOOT_ITERS = QUICK_BOOTSTRAP_ITERS if quick else BOOTSTRAP_ITERS

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

    variants_in = rag_results.get("variants", {})
    full_rows = variants_in.get("full_system", [])

    # Vertex generation scoring runs only for baseline + ablation groups (the
    # retrieval-isolation variants feed only the retrieval table, so their paid
    # generation scoring is skipped to bound cost).
    vertex_by_variant: Dict[str, Dict] = {}
    for name, rows in variants_in.items():
        group = VARIANT_CONFIGS.get(name, {}).get("group", "baseline")
        if group in ("baseline", "ablation"):
            print(f"\nScoring '{name}' ({group}) with Vertex AI...")
            vertex_by_variant[name] = _run_vertex_eval(rows, experiment_suffix=name)
        else:
            vertex_by_variant[name] = {"skipped": "retrieval-only variant"}

    full_vertex = vertex_by_variant.get("full_system", {})

    print("\nAggregating retrieval / generation / significance...")
    variants_out: Dict[str, Dict] = {}
    for name, rows in variants_in.items():
        group = VARIANT_CONFIGS.get(name, {}).get("group", "baseline")
        entry = {
            "group":           group,
            "retrieval":       _agg_retrieval(rows),
            "generation_extra": _agg_generation_extra(rows),
            "vertex_eval":     vertex_by_variant.get(name, {}),
        }
        if group == "ablation" and full_rows and name != "full_system":
            entry["significance_vs_full"] = _significance_vs_full(
                full_rows, rows, full_vertex, vertex_by_variant.get(name, {})
            )
        variants_out[name] = entry

    output = {
        "meta": {**rag_results.get("meta", {}), "quick": quick,
                 "bootstrap_iters": _BOOT_ITERS},
        "clarification_rate": rag_results.get("clarification_rate"),
        "ops": _agg_ops(rag_results.get("ops", {}), rag_results.get("clarification_rate")),
        "variants": variants_out,
    }

    with open(SCORES_PATH, "w") as f:
        json.dump(_nan_to_none(output), f, indent=2)
    print(f"\nScores saved to {SCORES_PATH}")

    fv = full_vertex.get("summary") if isinstance(full_vertex, dict) else None
    if fv:
        print("\n── full_system Vertex summary ──")
        for k, v in fv.items():
            print(f"  {k}: {v:.3f}")
    return output


def load_scores() -> Dict:
    with open(SCORES_PATH) as f:
        return json.load(f)
