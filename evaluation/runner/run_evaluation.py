"""
Main orchestrator for the Cryostat RAG evaluation pipeline.

Usage:
    python -m evaluation.runner.run_evaluation [options]

Options:
    --num-queries N         Number of queries per evaluation (default: from config)
    --skip-generation       Skip synthetic dataset generation (use cached)
    --ablation-only         Run only the ablation study
    --latency-only          Run only latency profiling
    --skip-bertscore        Skip BERTScore computation (faster)
    --pdf-folder PATH       Path to local PDF folder (skip GCS download)
    --output-dir PATH       Custom output directory
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import evaluation.env_setup  # noqa: F401 — must precede rag imports

from rag import (
    ContextExpandingHybridRetriever,
    ConversationHistory,
    VertexAIGeminiLLM,
    VertexAIEmbeddings,
    process_query,
    load_rag_if_available,
    initialize_rag_system,
    sync_vectorstore_from_gcs,
    load_and_split_pdfs,
    get_vectorstores,
    save_image_storage,
    save_bm25_docs,
    compute_manifest_from_local,
    save_manifest,
)

from evaluation.eval_config import (
    EVAL_NUM_QUERIES,
    ABLATION_QUERIES_PER_VARIANT,
    LATENCY_SAMPLE_SIZE,
    ABLATION_VARIANTS,
    SYNTHETIC_QA_PATH,
    OUTPUTS_DIR,
)
from evaluation.dataset.synthetic_generator import (
    generate_synthetic_dataset,
    load_synthetic_dataset,
)
from evaluation.metrics.retrieval_metrics import (
    compute_retrieval_metrics,
    compute_per_query_retrieval_metrics,
)
from evaluation.metrics.generation_metrics import (
    compute_generation_metrics,
    compute_per_query_generation_metrics,
)
from evaluation.metrics.llm_judge import compute_judge_metrics
from evaluation.metrics.custom_metrics import compute_all_custom_metrics
from evaluation.baselines.baseline_variants import run_variant, VARIANT_RUNNERS
from evaluation.analysis.statistics import compare_variants, summary_statistics, bootstrap_ci
from evaluation.analysis.latency_profiler import profile_queries
from evaluation.analysis.report_generator import (
    generate_markdown_report,
    generate_json_export,
    generate_charts,
)


def _init_rag(local_only: bool = False, pdf_folder: str = None) -> tuple:
    """Initialize the RAG system.

    Args:
        local_only: If True, skip all GCS vectorstore sync (both download and
            upload). If a local vectorstore already exists it is used as-is;
            otherwise it is built from `pdf_folder`. GCS is never touched, so
            the production vectorstore remains stable.
        pdf_folder: Local PDF directory used when `local_only=True` and no
            local vectorstore exists. Defaults to `.eval_data/pdfs`.
    """
    import evaluation.env_setup as _es  # noqa: F401 — get EVAL_LOCAL_STORAGE

    if local_only:
        print("Local-only mode: skipping GCS vectorstore sync.")
        retriever, llm, loaded = load_rag_if_available()
        if loaded:
            print("Loaded existing local vectorstore.")
        else:
            folder = pdf_folder or os.path.join(_es.EVAL_LOCAL_STORAGE, "pdfs")
            print(f"No local vectorstore found. Building from local PDFs at: {folder}")
            fine_docs, coarse_docs = load_and_split_pdfs(folder)
            fine_db, coarse_db = get_vectorstores(fine_docs, coarse_docs)
            save_image_storage()
            save_bm25_docs(fine_docs)
            manifest = compute_manifest_from_local(folder)
            save_manifest(manifest)
            # No GCS upload — local only
            retriever = ContextExpandingHybridRetriever(fine_db, coarse_db)
            llm = VertexAIGeminiLLM()
            print("Local vectorstore built successfully.")
    else:
        print("Loading RAG system...")
        sync_vectorstore_from_gcs()
        retriever, llm, loaded = load_rag_if_available()
        if not loaded:
            print("No local vectorstore found. Building from GCS...")
            retriever, llm = initialize_rag_system()
        else:
            print("Loaded existing vectorstore.")
    return retriever, llm


def _run_full_system_evaluation(
    queries: List[Dict],
    retriever,
    llm,
    embeddings,
    use_bertscore: bool = True,
) -> tuple:
    """Run the full system on all queries and collect results.

    Computes per-query metrics inline to avoid double LLM API calls.
    Returns per_query_log and per_category breakdown in addition to
    the aggregated retrieval_metrics and generation_metrics.
    """
    retrieval_results = []
    per_query_log = []

    # Accumulators for aggregate generation metrics
    faith_scores, relevancy_scores, sim_scores, bert_scores = [], [], [], []

    for q in tqdm(queries, desc="Evaluating full system"):
        conversation_history = ConversationHistory()
        query_text = q["question"]
        category = q.get("category", "unknown")

        # Retrieve documents
        docs = retriever.get_relevant_documents(query_text)
        source_chunks = q.get("source_chunks", [])
        retrieval_results.append({"retrieved_docs": docs, "source_chunks": source_chunks})

        # Generate answer
        try:
            result = process_query(query_text, retriever, llm, conversation_history, debug_mode=False)
            answer = result.get("answer", "")
        except Exception as e:
            answer = f"Error: {e}"

        # Per-query retrieval metrics (no LLM calls)
        per_ret = compute_per_query_retrieval_metrics(docs, source_chunks)

        # Per-query generation metrics (LLM calls — computed once, not twice)
        per_gen = compute_per_query_generation_metrics(
            query_text, answer, docs, llm, embeddings,
            reference_answer=q.get("reference_answer"),
            use_bertscore=use_bertscore,
        )

        faith_scores.append(per_gen.get("faithfulness", 0.0))
        relevancy_scores.append(per_gen.get("answer_relevancy", 0.0))
        if "semantic_similarity" in per_gen:
            sim_scores.append(per_gen["semantic_similarity"])
        if "bertscore_f1" in per_gen:
            bert_scores.append(per_gen["bertscore_f1"])

        per_query_log.append({
            "question_id": q.get("id", f"q_{len(per_query_log)}"),
            "category": category,
            "question": query_text,
            "reference_answer": q.get("reference_answer"),
            "generated_answer": answer,
            "retrieved_sources": [
                {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
                for d in docs
            ],
            **per_ret,
            **per_gen,
        })

    # Aggregate retrieval metrics
    retrieval_metrics = compute_retrieval_metrics(retrieval_results)

    # Aggregate generation metrics from per-query scores (avoids double LLM calls)
    generation_metrics: Dict = {
        "faithfulness": float(np.mean(faith_scores)) if faith_scores else 0.0,
        "answer_relevancy": float(np.mean(relevancy_scores)) if relevancy_scores else 0.0,
    }
    if sim_scores:
        generation_metrics["semantic_similarity"] = float(np.mean(sim_scores))
    if bert_scores:
        generation_metrics["bertscore_f1"] = float(np.mean(bert_scores))

    # Per-category breakdown
    numeric_keys = [
        "faithfulness", "answer_relevancy", "semantic_similarity", "bertscore_f1",
        "mrr", "precision@1", "recall@1", "hit_rate@1",
        "precision@5", "recall@5", "hit_rate@5",
        "precision@10", "recall@10", "hit_rate@10",
        "ndcg@10",
    ]
    per_category: Dict = {}
    for cat in sorted(set(e["category"] for e in per_query_log)):
        cat_entries = [e for e in per_query_log if e["category"] == cat]
        cat_metrics: Dict = {"n": len(cat_entries)}
        for key in numeric_keys:
            vals = [e[key] for e in cat_entries if key in e and isinstance(e[key], (int, float))]
            if vals:
                cat_metrics[key] = float(np.mean(vals))
        per_category[cat] = cat_metrics

    # Rebuild generation_results for the LLM judge (uses the logged data, no extra calls)
    generation_results = [
        {
            "query": e["question"],
            "answer": e["generated_answer"],
            "reference_answer": e["reference_answer"],
            "context_docs": retrieval_results[i]["retrieved_docs"],
        }
        for i, e in enumerate(per_query_log)
    ]

    return retrieval_metrics, generation_metrics, retrieval_results, generation_results, per_query_log, per_category


def _run_ablation_study(
    queries: List[Dict],
    retriever,
    llm,
    embeddings,
    variants: List[str] = None,
    use_bertscore: bool = False,
) -> Dict:
    """Run the ablation study across all variants."""
    if variants is None:
        variants = ABLATION_VARIANTS

    # Collect per-query scores for all variants
    full_system_scores = {}
    variant_scores = {}

    # Score names we'll track per query
    score_keys = ["faithfulness", "answer_relevancy", "semantic_similarity", "mrr"]

    # Run full system first
    print("\n--- Ablation: Full System ---")
    full_per_query = {k: [] for k in score_keys}

    for q in tqdm(queries, desc="Full system (ablation)"):
        conv = ConversationHistory()
        query_text = q["question"]

        try:
            result = process_query(query_text, retriever, llm, conv, debug_mode=False)
            answer = result.get("answer", "")
        except Exception:
            answer = ""

        docs = retriever.get_relevant_documents(query_text)

        # Per-query generation metrics
        gen = compute_per_query_generation_metrics(
            query_text, answer, docs, llm, embeddings,
            reference_answer=q.get("reference_answer"),
            use_bertscore=False,
        )
        for k in score_keys:
            if k in gen:
                full_per_query[k].append(gen[k])
            elif k == "mrr":
                from evaluation.metrics.retrieval_metrics import reciprocal_rank
                full_per_query[k].append(reciprocal_rank(docs, q.get("source_chunks", [])))

    full_system_scores = full_per_query

    # Run each variant
    for variant_name in variants:
        if variant_name == "full_system":
            continue

        print(f"\n--- Ablation: {variant_name} ---")
        v_per_query = {k: [] for k in score_keys}

        for q in tqdm(queries, desc=f"{variant_name}"):
            conv = ConversationHistory()
            query_text = q["question"]

            try:
                result = run_variant(variant_name, query_text, retriever, llm, conv)
                answer = result.get("answer", "")
            except Exception:
                answer = ""

            # Get docs for this variant
            if variant_name == "vanilla_llm":
                docs = []
            else:
                docs = retriever.get_relevant_documents(query_text)

            gen = compute_per_query_generation_metrics(
                query_text, answer, docs, llm, embeddings,
                reference_answer=q.get("reference_answer"),
                use_bertscore=False,
            )
            for k in score_keys:
                if k in gen:
                    v_per_query[k].append(gen[k])
                elif k == "mrr":
                    from evaluation.metrics.retrieval_metrics import reciprocal_rank
                    v_per_query[k].append(reciprocal_rank(docs, q.get("source_chunks", [])))

        variant_scores[variant_name] = v_per_query

    # Statistical comparison
    comparisons = compare_variants(full_system_scores, variant_scores)
    return comparisons


def _run_conversational_evaluation(
    all_queries: List[Dict],
    retriever,
    llm,
) -> Dict:
    """Evaluate multi-turn conversational chains.

    Groups questions that share a conversation_id and runs them sequentially
    with a shared ConversationHistory, so follow-up questions receive real context.
    Returns per-conversation and aggregate results.
    """
    from collections import defaultdict
    from rag import is_follow_up_query

    conv_groups: Dict[str, List[Dict]] = defaultdict(list)
    for q in all_queries:
        if "conversation_id" in q:
            conv_groups[q["conversation_id"]].append(q)

    if not conv_groups:
        return {"note": "No conversation_id fields found in dataset."}

    # Sort each group by turn_order
    for cid in conv_groups:
        conv_groups[cid].sort(key=lambda x: x.get("turn_order", 0))

    conv_results = {}
    all_follow_up_correct = []

    for conv_id, turns in conv_groups.items():
        history = ConversationHistory()
        turn_log = []

        for turn in turns:
            query_text = turn["question"]
            turn_num = turn.get("turn_order", 0)
            expected = turn.get("expected_behavior", "")

            try:
                result = process_query(query_text, retriever, llm, history, debug_mode=False)
                answer = result.get("answer", "")
            except Exception as e:
                answer = f"Error: {e}"

            detected_followup = is_follow_up_query(query_text)
            correct = (expected == "follow_up") == detected_followup
            if expected == "follow_up":
                all_follow_up_correct.append(correct)

            turn_log.append({
                "turn": turn_num,
                "question": query_text,
                "answer": answer[:300],
                "expected_behavior": expected,
                "detected_as_followup": detected_followup,
                "followup_detection_correct": correct,
            })

        conv_results[conv_id] = {"turns": turn_log}

    followup_accuracy = float(np.mean(all_follow_up_correct)) if all_follow_up_correct else None
    return {
        "conversations": conv_results,
        "followup_detection_accuracy_with_context": followup_accuracy,
        "n_conversations": len(conv_groups),
        "n_followup_turns_evaluated": len(all_follow_up_correct),
    }


def run_evaluation(args):
    """Main evaluation pipeline."""
    start_time = time.time()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Initialize
    retriever, llm = _init_rag(
        local_only=args.local_only,
        pdf_folder=args.pdf_folder,
    )
    embeddings = VertexAIEmbeddings()

    # Step 1: Dataset
    if args.skip_generation and os.path.exists(SYNTHETIC_QA_PATH):
        print("Loading cached synthetic dataset...")
        all_queries = load_synthetic_dataset()
    else:
        print("Generating synthetic dataset...")
        all_queries = generate_synthetic_dataset(pdf_folder=args.pdf_folder)

    dataset_summary = {
        "total": len(all_queries),
        "categories": {},
    }
    for q in all_queries:
        cat = q.get("category", "unknown")
        dataset_summary["categories"][cat] = dataset_summary["categories"].get(cat, 0) + 1

    print(f"\nDataset: {len(all_queries)} questions")
    for cat, count in dataset_summary["categories"].items():
        print(f"  {cat}: {count}")

    # Filter queries for evaluation
    eval_queries = [q for q in all_queries if q.get("category") != "edge_case"]
    edge_queries = [q for q in all_queries if q.get("category") == "edge_case"]

    if args.num_queries and args.num_queries < len(eval_queries):
        eval_queries = eval_queries[:args.num_queries]

    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_summary": dataset_summary,
    }

    # Latency-only mode
    if args.latency_only:
        print("\n=== Latency Profiling ===")
        latency_queries = [q["question"] for q in eval_queries[:LATENCY_SAMPLE_SIZE]]
        latency_results = profile_queries(
            latency_queries, retriever, llm,
            status_callback=lambda i, n: print(f"  Profiling query {i}/{n}", end="\r"),
        )
        results["latency"] = latency_results
        generate_json_export(results)
        generate_charts({}, latency_results)
        print(f"\nDone in {time.time() - start_time:.1f}s")
        return results

    # Ablation-only mode
    if args.ablation_only:
        print("\n=== Ablation Study ===")
        ablation_queries = eval_queries[:ABLATION_QUERIES_PER_VARIANT]
        ablation_comparisons = _run_ablation_study(
            ablation_queries, retriever, llm, embeddings,
            use_bertscore=not args.skip_bertscore,
        )
        results["ablation"] = ablation_comparisons
        generate_markdown_report(
            retrieval_metrics={}, generation_metrics={}, judge_metrics={},
            custom_metrics={}, ablation_comparisons=ablation_comparisons,
            latency_results={}, dataset_summary=dataset_summary,
        )
        generate_json_export(results)
        generate_charts(ablation_comparisons, {})
        print(f"\nDone in {time.time() - start_time:.1f}s")
        return results

    # Full evaluation pipeline
    # Step 2: Retrieval + Generation evaluation
    print("\n=== Full System Evaluation ===")
    retrieval_metrics, generation_metrics, retrieval_results, generation_results, per_query_log, per_category = (
        _run_full_system_evaluation(
            eval_queries, retriever, llm, embeddings,
            use_bertscore=not args.skip_bertscore,
        )
    )
    results["retrieval_metrics"] = retrieval_metrics
    results["generation_metrics"] = generation_metrics
    results["per_category"] = per_category
    print(f"Retrieval: {retrieval_metrics}")
    print(f"Generation: {generation_metrics}")
    print("Per-category breakdown:")
    for cat, metrics in per_category.items():
        print(f"  {cat} (n={metrics.get('n')}): mrr={metrics.get('mrr', 0):.3f}, "
              f"faithfulness={metrics.get('faithfulness', 0):.3f}, "
              f"answer_relevancy={metrics.get('answer_relevancy', 0):.3f}")

    # Save per-query log alongside main results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    per_query_path = os.path.join(OUTPUTS_DIR, f"per_query_results_{timestamp}.json")
    with open(per_query_path, "w") as f:
        json.dump(per_query_log, f, indent=2, default=str)
    print(f"Per-query results saved to: {per_query_path}")

    # Step 3: LLM Judge
    print("\n=== LLM Judge Evaluation ===")
    judge_metrics = compute_judge_metrics(generation_results, llm, with_consistency=False)
    results["judge_metrics"] = judge_metrics
    print(f"Judge: {judge_metrics}")

    # Step 4: Custom metrics
    print("\n=== Custom Feature Metrics ===")
    custom_metrics = compute_all_custom_metrics(all_queries, retriever)
    results["custom_metrics"] = custom_metrics
    for feature, metrics in custom_metrics.items():
        print(f"  {feature}: {metrics}")

    # Step 4b: Conversational evaluation (multi-turn chains with real history)
    print("\n=== Conversational Evaluation ===")
    conv_eval = _run_conversational_evaluation(all_queries, retriever, llm)
    results["conversational_eval"] = conv_eval
    if conv_eval.get("followup_detection_accuracy_with_context") is not None:
        print(f"  Follow-up detection accuracy (with real context): "
              f"{conv_eval['followup_detection_accuracy_with_context']:.3f}")
    print(f"  Conversations evaluated: {conv_eval.get('n_conversations', 0)}")

    # Step 5: Ablation study
    print("\n=== Ablation Study ===")
    ablation_queries = eval_queries[:ABLATION_QUERIES_PER_VARIANT]
    ablation_comparisons = _run_ablation_study(
        ablation_queries, retriever, llm, embeddings,
        use_bertscore=not args.skip_bertscore,
    )
    results["ablation"] = ablation_comparisons

    # Step 6: Latency profiling
    print("\n=== Latency Profiling ===")
    latency_queries = [q["question"] for q in eval_queries[:LATENCY_SAMPLE_SIZE]]
    latency_results = profile_queries(
        latency_queries, retriever, llm,
        status_callback=lambda i, n: print(f"  Profiling query {i}/{n}", end="\r"),
    )
    results["latency"] = latency_results

    # Step 7: Generate outputs
    print("\n=== Generating Report ===")
    generate_markdown_report(
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        judge_metrics=judge_metrics,
        custom_metrics=custom_metrics,
        ablation_comparisons=ablation_comparisons,
        latency_results=latency_results,
        dataset_summary=dataset_summary,
    )
    generate_json_export(results)
    generate_charts(
        ablation_comparisons, latency_results,
        retrieval_metrics, generation_metrics,
    )

    elapsed = time.time() - start_time
    print(f"\nEvaluation complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Cryostat RAG Evaluation Pipeline")
    parser.add_argument("--num-queries", type=int, default=None,
                        help=f"Number of queries to evaluate (default: all)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Use cached synthetic dataset")
    parser.add_argument("--ablation-only", action="store_true",
                        help="Run only the ablation study")
    parser.add_argument("--latency-only", action="store_true",
                        help="Run only latency profiling")
    parser.add_argument("--skip-bertscore", action="store_true",
                        help="Skip BERTScore (faster)")
    parser.add_argument("--pdf-folder", type=str, default=None,
                        help="Local PDF folder path (skip GCS download)")
    parser.add_argument("--local-only", action="store_true",
                        help="Skip all GCS vectorstore sync; build from --pdf-folder if needed. "
                             "GCS production vectorstore is never touched.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory")

    args = parser.parse_args()

    if args.output_dir:
        import evaluation.eval_config as cfg
        cfg.OUTPUTS_DIR = args.output_dir

    run_evaluation(args)


if __name__ == "__main__":
    main()
