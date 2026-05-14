"""
Report generator for the evaluation framework.

Produces publication-ready markdown reports and matplotlib charts.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from evaluation.eval_config import OUTPUTS_DIR, CHART_DPI, CHART_STYLE, ABLATION_VARIANTS
from evaluation.analysis.latency_profiler import format_latency_table


def _ensure_output_dir():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def generate_markdown_report(
    retrieval_metrics: Dict,
    generation_metrics: Dict,
    judge_metrics: Dict,
    custom_metrics: Dict,
    ablation_comparisons: Dict,
    latency_results: Dict,
    dataset_summary: Dict,
    output_path: str = None,
) -> str:
    """Generate a comprehensive markdown evaluation report."""
    _ensure_output_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if output_path is None:
        output_path = os.path.join(OUTPUTS_DIR, f"evaluation_report_{timestamp}.md")

    sections = []

    # Header
    sections.append("# Cryostat RAG Evaluation Report")
    sections.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Dataset summary
    sections.append("## 1. Evaluation Dataset\n")
    if dataset_summary:
        sections.append(f"- Total questions: {dataset_summary.get('total', 0)}")
        for cat, count in dataset_summary.get("categories", {}).items():
            sections.append(f"  - {cat}: {count}")
        sections.append("")

    # Retrieval metrics
    sections.append("## 2. Retrieval Quality\n")
    if retrieval_metrics:
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        for metric, value in sorted(retrieval_metrics.items()):
            sections.append(f"| {metric} | {value:.4f} |")
        sections.append("")

    # Generation metrics
    sections.append("## 3. Generation Quality\n")
    if generation_metrics:
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        for metric, value in sorted(generation_metrics.items()):
            sections.append(f"| {metric} | {value:.4f} |")
        sections.append("")

    # LLM Judge scores
    sections.append("## 4. LLM Judge Scores\n")
    if judge_metrics:
        sections.append("| Dimension | Score (1-5) |")
        sections.append("|-----------|-------------|")
        for metric, value in sorted(judge_metrics.items()):
            sections.append(f"| {metric} | {value:.2f} |")
        sections.append("")

    # Custom metrics
    sections.append("## 5. Feature-Specific Metrics\n")
    if custom_metrics:
        for feature, feature_metrics in custom_metrics.items():
            sections.append(f"### {feature.replace('_', ' ').title()}\n")
            sections.append("| Metric | Value |")
            sections.append("|--------|-------|")
            for metric, value in sorted(feature_metrics.items()):
                if isinstance(value, float):
                    sections.append(f"| {metric} | {value:.4f} |")
                else:
                    sections.append(f"| {metric} | {value} |")
            sections.append("")

    # Ablation study
    sections.append("## 6. Feature Ablation Study\n")
    sections.append("Delta = Full System - Variant (positive = feature helps)\n")
    if ablation_comparisons:
        for variant, metrics in ablation_comparisons.items():
            sections.append(f"### {variant.replace('_', ' ').title()}\n")
            sections.append("| Metric | Full | Variant | Delta | 95% CI | p-value | Effect |")
            sections.append("|--------|------|---------|-------|--------|---------|--------|")
            for metric, data in sorted(metrics.items()):
                ci = f"[{data['delta_ci_lower']:.3f}, {data['delta_ci_upper']:.3f}]"
                sig = "*" if data["significant"] else ""
                sections.append(
                    f"| {metric} | {data['full_system_mean']:.3f} | "
                    f"{data['variant_mean']:.3f} | {data['delta']:+.3f}{sig} | "
                    f"{ci} | {data['p_value']:.4f} | {data['effect_interpretation']} |"
                )
            sections.append("")
        sections.append("\\* Statistically significant at p < 0.05\n")

    # Latency
    sections.append("## 7. Latency Analysis\n")
    if latency_results:
        sections.append(format_latency_table(latency_results))
        sections.append("")

    # Methodology notes
    sections.append("## 8. Methodology Notes\n")
    sections.append("- **Retrieval metrics**: Precision@K, Recall@K, Hit Rate@K, NDCG@K, MRR")
    sections.append("- **Generation metrics**: RAGAS Faithfulness, Answer Relevancy, Semantic Similarity, BERTScore F1")
    sections.append("- **LLM Judge**: Gemini 2.5 Pro with structured rubric (1-5 scale)")
    sections.append("- **Statistical tests**: Bootstrap CI (1000 iterations), Wilcoxon signed-rank / paired t-test")
    sections.append("- **Effect size**: Cohen's d with interpretation (negligible/small/medium/large)")
    sections.append("- **Ablation**: Each feature disabled independently; same queries used across all variants")
    sections.append("")

    report = "\n".join(sections)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to {output_path}")
    return output_path


def generate_json_export(
    all_results: Dict,
    output_path: str = None,
) -> str:
    """Export all raw results as JSON for further analysis."""
    _ensure_output_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if output_path is None:
        output_path = os.path.join(OUTPUTS_DIR, f"evaluation_results_{timestamp}.json")

    # Make numpy types JSON-serializable
    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"Results exported to {output_path}")
    return output_path


def generate_charts(
    ablation_comparisons: Dict,
    latency_results: Dict,
    retrieval_metrics: Dict = None,
    generation_metrics: Dict = None,
    output_dir: str = None,
) -> List[str]:
    """Generate matplotlib charts for the report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            plt.style.use(CHART_STYLE)
        except OSError:
            plt.style.use("seaborn-v0_8")
    except ImportError:
        print("Warning: matplotlib not installed. Skipping chart generation.")
        return []

    _ensure_output_dir()
    if output_dir is None:
        output_dir = os.path.join(OUTPUTS_DIR, "charts")
    os.makedirs(output_dir, exist_ok=True)

    chart_paths = []

    # 1. Ablation comparison bar chart
    if ablation_comparisons:
        # Collect metrics across variants
        all_metrics = set()
        for variant_data in ablation_comparisons.values():
            all_metrics.update(variant_data.keys())

        # Pick key metrics to display
        key_metrics = [m for m in ["faithfulness", "answer_relevancy", "mrr", "judge_overall"] if m in all_metrics]
        if not key_metrics:
            key_metrics = list(all_metrics)[:4]

        for metric in key_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            variants = []
            deltas = []
            ci_lower = []
            ci_upper = []
            colors = []

            for variant, metrics_data in ablation_comparisons.items():
                if metric in metrics_data:
                    data = metrics_data[metric]
                    variants.append(variant.replace("_", "\n"))
                    deltas.append(data["delta"])
                    ci_lower.append(data["delta"] - data["delta_ci_lower"])
                    ci_upper.append(data["delta_ci_upper"] - data["delta"])
                    colors.append("#2ecc71" if data["significant"] else "#95a5a6")

            if variants:
                x = range(len(variants))
                bars = ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.5)
                ax.errorbar(x, deltas, yerr=[ci_lower, ci_upper], fmt="none", color="black", capsize=4)
                ax.set_xticks(x)
                ax.set_xticklabels(variants, fontsize=9)
                ax.set_ylabel(f"Delta ({metric})")
                ax.set_title(f"Feature Ablation: {metric}\n(Full System - Variant, positive = feature helps)")
                ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
                ax.legend(
                    [plt.Rectangle((0, 0), 1, 1, fc="#2ecc71"), plt.Rectangle((0, 0), 1, 1, fc="#95a5a6")],
                    ["Significant (p<0.05)", "Not significant"],
                    loc="upper right",
                )

                path = os.path.join(output_dir, f"ablation_{metric}.png")
                fig.tight_layout()
                fig.savefig(path, dpi=CHART_DPI)
                plt.close(fig)
                chart_paths.append(path)

    # 2. Latency breakdown
    if latency_results:
        stages = [s for s in ["retrieval", "context_expansion", "llm_generation", "prompt_assembly",
                               "followup_detection", "doc_type_detection", "image_extraction"]
                  if s in latency_results]

        if stages:
            fig, ax = plt.subplots(figsize=(10, 6))
            means = [latency_results[s].get("mean", 0) for s in stages]
            labels = [s.replace("_", "\n") for s in stages]

            bars = ax.barh(range(len(stages)), means, color="#3498db", edgecolor="black", linewidth=0.5)
            ax.set_yticks(range(len(stages)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel("Mean Latency (seconds)")
            ax.set_title("Pipeline Component Latency Breakdown")

            # Add value labels
            for bar, val in zip(bars, means):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}s", va="center", fontsize=9)

            path = os.path.join(output_dir, "latency_breakdown.png")
            fig.tight_layout()
            fig.savefig(path, dpi=CHART_DPI)
            plt.close(fig)
            chart_paths.append(path)

    # 3. Metrics overview radar chart
    if retrieval_metrics and generation_metrics:
        combined = {}
        for k, v in retrieval_metrics.items():
            if "@" not in k or k.endswith("@5"):
                combined[k] = v
        combined.update(generation_metrics)

        if len(combined) >= 3:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            labels = list(combined.keys())
            values = list(combined.values())
            values.append(values[0])  # Close the polygon
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            angles.append(angles[0])

            ax.fill(angles, values, alpha=0.25, color="#3498db")
            ax.plot(angles, values, color="#3498db", linewidth=2)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title("Metrics Overview", pad=20)

            path = os.path.join(output_dir, "metrics_radar.png")
            fig.tight_layout()
            fig.savefig(path, dpi=CHART_DPI)
            plt.close(fig)
            chart_paths.append(path)

    print(f"Generated {len(chart_paths)} charts in {output_dir}")
    return chart_paths
