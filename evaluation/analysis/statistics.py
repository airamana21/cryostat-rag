"""
Statistical analysis utilities for the evaluation framework.

Provides bootstrap confidence intervals, paired statistical tests,
and effect size calculations for rigorous metric comparison.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats

from evaluation.eval_config import CONFIDENCE_LEVEL, BOOTSTRAP_ITERATIONS, SIGNIFICANCE_THRESHOLD


def bootstrap_ci(
    values: List[float],
    confidence: float = CONFIDENCE_LEVEL,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    statistic: str = "mean",
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: Sample values.
        confidence: Confidence level (e.g. 0.95).
        n_iterations: Number of bootstrap resamples.
        statistic: "mean" or "median".

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    if not values:
        return (0.0, 0.0, 0.0)

    arr = np.array(values)
    stat_fn = np.mean if statistic == "mean" else np.median
    point_estimate = float(stat_fn(arr))

    bootstrap_stats = []
    rng = np.random.default_rng(42)
    for _ in range(n_iterations):
        resample = rng.choice(arr, size=len(arr), replace=True)
        bootstrap_stats.append(stat_fn(resample))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return (point_estimate, lower, upper)


def paired_test(
    values_a: List[float],
    values_b: List[float],
    test: str = "auto",
) -> Dict:
    """Run a paired statistical test comparing two sets of scores.

    Uses Wilcoxon signed-rank if sample < 30 or non-normal, else paired t-test.

    Args:
        values_a: Scores from system A (e.g. full system).
        values_b: Scores from system B (e.g. ablation variant).
        test: "ttest", "wilcoxon", or "auto".

    Returns:
        Dict with test name, statistic, p-value, and significance.
    """
    a = np.array(values_a)
    b = np.array(values_b)

    if len(a) != len(b):
        raise ValueError(f"Arrays must have same length: {len(a)} vs {len(b)}")

    n = len(a)
    if n < 3:
        return {"test": "insufficient_data", "p_value": 1.0, "significant": False, "n": n}

    # Choose test
    if test == "auto":
        # Use Wilcoxon for small samples or non-normal differences
        diffs = a - b
        if n < 30:
            test = "wilcoxon"
        else:
            _, normality_p = stats.shapiro(diffs)
            test = "ttest" if normality_p > 0.05 else "wilcoxon"

    if test == "ttest":
        statistic, p_value = stats.ttest_rel(a, b)
        test_name = "paired_t_test"
    else:
        # Wilcoxon needs non-zero differences
        diffs = a - b
        if np.all(diffs == 0):
            return {"test": "wilcoxon", "statistic": 0.0, "p_value": 1.0, "significant": False, "n": n}
        statistic, p_value = stats.wilcoxon(diffs)
        test_name = "wilcoxon_signed_rank"

    return {
        "test": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < SIGNIFICANCE_THRESHOLD,
        "n": n,
    }


def cohens_d(values_a: List[float], values_b: List[float]) -> float:
    """Compute Cohen's d effect size for paired samples."""
    a = np.array(values_a)
    b = np.array(values_b)
    diffs = a - b
    if np.std(diffs) == 0:
        return 0.0
    return float(np.mean(diffs) / np.std(diffs, ddof=1))


def effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def compare_variants(
    full_system_scores: Dict[str, List[float]],
    variant_scores: Dict[str, Dict[str, List[float]]],
) -> Dict:
    """Compare the full system against each ablation variant.

    Args:
        full_system_scores: Dict of metric_name -> list of per-query scores.
        variant_scores: Dict of variant_name -> {metric_name -> list of per-query scores}.

    Returns:
        Nested dict: variant -> metric -> {delta, ci, p_value, effect_size, ...}
    """
    comparisons = {}

    for variant_name, v_scores in variant_scores.items():
        comparisons[variant_name] = {}

        for metric_name, full_values in full_system_scores.items():
            variant_values = v_scores.get(metric_name, [])

            if not full_values or not variant_values:
                continue

            # Ensure same length (paired)
            min_len = min(len(full_values), len(variant_values))
            fv = full_values[:min_len]
            vv = variant_values[:min_len]

            # Point estimates
            full_mean = float(np.mean(fv))
            variant_mean = float(np.mean(vv))
            delta = full_mean - variant_mean

            # Bootstrap CI on the delta
            deltas = [a - b for a, b in zip(fv, vv)]
            _, delta_lower, delta_upper = bootstrap_ci(deltas)

            # Statistical test
            test_result = paired_test(fv, vv)

            # Effect size
            d = cohens_d(fv, vv)

            comparisons[variant_name][metric_name] = {
                "full_system_mean": full_mean,
                "variant_mean": variant_mean,
                "delta": delta,
                "delta_ci_lower": delta_lower,
                "delta_ci_upper": delta_upper,
                "p_value": test_result["p_value"],
                "significant": test_result["significant"],
                "test_used": test_result["test"],
                "cohens_d": d,
                "effect_interpretation": effect_size_interpretation(d),
                "n": min_len,
            }

    return comparisons


def summary_statistics(values: List[float]) -> Dict:
    """Compute summary statistics for a list of values."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}

    arr = np.array(values)
    mean, lower, upper = bootstrap_ci(values)

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci_lower": lower,
        "ci_upper": upper,
        "n": len(arr),
    }
