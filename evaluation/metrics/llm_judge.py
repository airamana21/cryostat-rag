"""
LLM-as-judge evaluation module.

Uses Gemini to score RAG responses on a structured rubric (1-5 scale)
across multiple dimensions. Includes calibration and consistency checks.
"""
import json
import os
import sys
from typing import List, Dict, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import evaluation.env_setup  # noqa: F401 — must precede rag imports

from rag import VertexAIGeminiLLM
from evaluation.eval_config import (
    JUDGE_DIMENSIONS,
    JUDGE_RUNS_PER_QUERY,
    JUDGE_SCORE_SCALE,
)

JUDGE_PROMPT = """You are an expert evaluator for a cryogenic equipment support chatbot. Your task is to evaluate the quality of a response on a scale of {min_score}-{max_score}.

## Question
{question}

## Retrieved Context (from source documents)
{context}

## Generated Response
{response}

{reference_section}

## Evaluation Criteria

Score each dimension from {min_score} (worst) to {max_score} (best):

1. **Factual Accuracy** ({min_score}-{max_score}): Are the facts in the response correct based on the source documents? Deduct for any incorrect claims.
2. **Completeness** ({min_score}-{max_score}): Does the response fully address the question? Does it cover all relevant aspects?
3. **Clarity** ({min_score}-{max_score}): Is the response well-organized, easy to read, and logically structured?
4. **Source Grounding** ({min_score}-{max_score}): Does the response stick to information from the retrieved context? Deduct for hallucinated information not in the context.

## Instructions
For each dimension, first provide a brief reasoning (1-2 sentences), then give your score.

Respond in this exact JSON format (no markdown, no code fences):
{{
    "factual_accuracy": {{"reasoning": "...", "score": N}},
    "completeness": {{"reasoning": "...", "score": N}},
    "clarity": {{"reasoning": "...", "score": N}},
    "source_grounding": {{"reasoning": "...", "score": N}}
}}"""


def _parse_judge_response(response: str) -> Optional[Dict]:
    """Parse the judge's JSON response."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None


def judge_response(
    question: str,
    response: str,
    context_docs: List,
    llm: VertexAIGeminiLLM,
    reference_answer: Optional[str] = None,
) -> Optional[Dict]:
    """Score a single response using the LLM judge.

    Returns dict with scores and reasoning for each dimension,
    plus an overall score (average).
    """
    min_score, max_score = JUDGE_SCORE_SCALE

    context_text = "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')} p.{doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in context_docs
    ) if context_docs else "(No context retrieved)"

    reference_section = ""
    if reference_answer:
        reference_section = f"## Reference Answer (ground truth)\n{reference_answer}\n"

    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context_text,
        response=response,
        reference_section=reference_section,
        min_score=min_score,
        max_score=max_score,
    )

    try:
        raw = llm._call(prompt)
        parsed = _parse_judge_response(raw)
        if not parsed:
            return None
    except Exception as e:
        print(f"Judge error: {e}")
        return None

    # Extract scores
    result = {}
    scores = []
    for dim in JUDGE_DIMENSIONS:
        dim_data = parsed.get(dim, {})
        score = dim_data.get("score", min_score)
        score = max(min_score, min(max_score, int(score)))
        reasoning = dim_data.get("reasoning", "")
        result[dim] = {"score": score, "reasoning": reasoning}
        scores.append(score)

    result["overall"] = float(np.mean(scores))
    return result


def judge_with_consistency(
    question: str,
    response: str,
    context_docs: List,
    llm: VertexAIGeminiLLM,
    reference_answer: Optional[str] = None,
    n_runs: int = JUDGE_RUNS_PER_QUERY,
) -> Dict:
    """Run the judge multiple times and compute consistency metrics.

    Returns aggregated scores with mean, std, and per-run details.
    """
    all_runs = []
    for _ in range(n_runs):
        result = judge_response(question, response, context_docs, llm, reference_answer)
        if result:
            all_runs.append(result)

    if not all_runs:
        return {
            "scores": {dim: {"mean": 0.0, "std": 0.0} for dim in JUDGE_DIMENSIONS},
            "overall_mean": 0.0,
            "overall_std": 0.0,
            "n_successful_runs": 0,
            "consistency": 0.0,
        }

    # Aggregate per dimension
    aggregated = {}
    for dim in JUDGE_DIMENSIONS:
        dim_scores = [run[dim]["score"] for run in all_runs if dim in run]
        aggregated[dim] = {
            "mean": float(np.mean(dim_scores)) if dim_scores else 0.0,
            "std": float(np.std(dim_scores)) if dim_scores else 0.0,
        }

    overall_scores = [run["overall"] for run in all_runs]

    # Consistency: 1 - (mean std across dimensions / score range)
    mean_std = np.mean([aggregated[dim]["std"] for dim in JUDGE_DIMENSIONS])
    score_range = JUDGE_SCORE_SCALE[1] - JUDGE_SCORE_SCALE[0]
    consistency = 1.0 - (mean_std / score_range) if score_range > 0 else 0.0

    return {
        "scores": aggregated,
        "overall_mean": float(np.mean(overall_scores)),
        "overall_std": float(np.std(overall_scores)),
        "n_successful_runs": len(all_runs),
        "consistency": float(consistency),
        "runs": all_runs,
    }


def compute_judge_metrics(
    results: List[Dict],
    llm: VertexAIGeminiLLM,
    with_consistency: bool = False,
) -> Dict:
    """Compute LLM judge scores across all evaluation results.

    Args:
        results: List of dicts with query, answer, context_docs, reference_answer.
        llm: LLM to use as judge.
        with_consistency: Run multiple times per query for consistency analysis.

    Returns:
        Dict with average scores per dimension and overall.
    """
    all_scores = {dim: [] for dim in JUDGE_DIMENSIONS}
    all_overall = []
    consistency_scores = []

    for r in results:
        if with_consistency:
            result = judge_with_consistency(
                r["query"], r["answer"], r.get("context_docs", []),
                llm, r.get("reference_answer"),
            )
            for dim in JUDGE_DIMENSIONS:
                all_scores[dim].append(result["scores"][dim]["mean"])
            all_overall.append(result["overall_mean"])
            consistency_scores.append(result["consistency"])
        else:
            result = judge_response(
                r["query"], r["answer"], r.get("context_docs", []),
                llm, r.get("reference_answer"),
            )
            if result:
                for dim in JUDGE_DIMENSIONS:
                    all_scores[dim].append(result[dim]["score"])
                all_overall.append(result["overall"])

    metrics = {}
    for dim in JUDGE_DIMENSIONS:
        if all_scores[dim]:
            metrics[f"judge_{dim}"] = float(np.mean(all_scores[dim]))
    if all_overall:
        metrics["judge_overall"] = float(np.mean(all_overall))
    if consistency_scores:
        metrics["judge_consistency"] = float(np.mean(consistency_scores))

    return metrics


def calibrate_judge(
    gold_standard: List[Dict],
    llm: VertexAIGeminiLLM,
) -> Dict:
    """Calibrate the judge against gold standard human ratings.

    Args:
        gold_standard: List of dicts with query, answer, context_docs,
            reference_answer, and human_scores (dict of dimension -> score).

    Returns:
        Calibration metrics including Spearman correlation per dimension.
    """
    from scipy.stats import spearmanr

    judge_scores = {dim: [] for dim in JUDGE_DIMENSIONS}
    human_scores = {dim: [] for dim in JUDGE_DIMENSIONS}

    for item in gold_standard:
        result = judge_response(
            item["query"], item["answer"], item.get("context_docs", []),
            llm, item.get("reference_answer"),
        )
        if result and "human_scores" in item:
            for dim in JUDGE_DIMENSIONS:
                if dim in item["human_scores"] and dim in result:
                    judge_scores[dim].append(result[dim]["score"])
                    human_scores[dim].append(item["human_scores"][dim])

    calibration = {}
    for dim in JUDGE_DIMENSIONS:
        if len(judge_scores[dim]) >= 3:
            rho, p_value = spearmanr(judge_scores[dim], human_scores[dim])
            calibration[dim] = {
                "spearman_rho": float(rho),
                "p_value": float(p_value),
                "n_samples": len(judge_scores[dim]),
            }
        else:
            calibration[dim] = {
                "spearman_rho": None,
                "p_value": None,
                "n_samples": len(judge_scores[dim]),
                "note": "Insufficient samples for correlation",
            }

    return calibration
