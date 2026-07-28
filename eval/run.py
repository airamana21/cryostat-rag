"""
End-to-end evaluation runner for the cryostat RAG system.

Usage:
    python eval/run.py --all                  # run every step
    python eval/run.py --quick                # fast smoke test (subset of Qs + variants)
    python eval/run.py --dataset              # step 1: generate Q&A dataset from GCS PDFs
    python eval/run.py --pipeline             # step 2: run RAG, collect answers
    python eval/run.py --pipeline --no-ablation   # skip ablation variants
    python eval/run.py --pipeline --skip-rebuild  # reuse existing local vectorstore
    python eval/run.py --score                # step 3: Vertex AI scoring + retrieval metrics
    python eval/run.py --export               # step 4: LaTeX tables + charts

Prerequisites:
    pip install -r eval/eval_requirements.txt
    gcloud auth application-default login     # or set GOOGLE_APPLICATION_CREDENTIALS
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import DATASET_PATH, RAG_RESULTS_PATH, SCORES_PATH, OUTPUTS_DIR


def _step_dataset(args):
    from eval.dataset import generate_dataset
    pdf_folder = getattr(args, "pdf_folder", None)
    print("=" * 60)
    print("STEP 1: Generating synthetic Q&A dataset")
    print("=" * 60)
    t0 = time.time()
    dataset = generate_dataset(pdf_folder=pdf_folder)
    print(f"\nDone in {time.time() - t0:.1f}s — {len(dataset)} questions")
    return dataset


def _quick_slice(dataset):
    """Small, representative subset for smoke tests: a few gold-chunk questions
    (so retrieval metrics compute) plus a couple of edge cases (clarification)."""
    from eval.config import QUICK_N_QUESTIONS
    has_gold = lambda it: bool(it.get("gold_chunks") or it.get("source_chunk_text"))
    gold = [it for it in dataset if has_gold(it)]
    edge = [it for it in dataset if not has_gold(it)]
    return gold[:QUICK_N_QUESTIONS] + edge[:2]


def _step_pipeline(args, dataset=None):
    from eval.pipeline import run_pipeline
    from eval.config import VARIANT_CONFIGS, QUICK_VARIANTS

    if dataset is None:
        if not os.path.exists(DATASET_PATH):
            print(f"ERROR: {DATASET_PATH} not found. Run --dataset first.")
            sys.exit(1)
        with open(DATASET_PATH) as f:
            dataset = json.load(f)

    quick = getattr(args, "quick", False)
    if quick:
        dataset = _quick_slice(dataset)
        variants = QUICK_VARIANTS
    elif getattr(args, "no_ablation", False):
        variants = [v for v, c in VARIANT_CONFIGS.items() if c["group"] != "ablation"]
    else:
        variants = None  # all variants

    print("=" * 60)
    print("STEP 2: Running RAG pipeline" + (" (quick)" if quick else ""))
    print("=" * 60)
    t0 = time.time()
    results = run_pipeline(
        dataset,
        variants=variants,
        skip_rebuild=getattr(args, "skip_rebuild", False) or quick,
        quick=quick,
    )
    print(f"\nDone in {time.time() - t0:.1f}s")
    return results


def _step_score(args, rag_results=None):
    from eval.score import score, load_scores

    if rag_results is None:
        if not os.path.exists(RAG_RESULTS_PATH):
            print(f"ERROR: {RAG_RESULTS_PATH} not found. Run --pipeline first.")
            sys.exit(1)
        with open(RAG_RESULTS_PATH) as f:
            rag_results = json.load(f)

    print("=" * 60)
    print("STEP 3: Scoring with Vertex AI Evaluation SDK")
    print("=" * 60)
    t0 = time.time()
    scores = score(rag_results, quick=getattr(args, "quick", False))
    print(f"\nDone in {time.time() - t0:.1f}s")
    return scores


def _step_export(args, scores=None):
    from eval.export import export

    if scores is None:
        if not os.path.exists(SCORES_PATH):
            print(f"ERROR: {SCORES_PATH} not found. Run --score first.")
            sys.exit(1)
        with open(SCORES_PATH) as f:
            scores = json.load(f)

    print("=" * 60)
    print("STEP 4: Exporting paper-quality outputs")
    print("=" * 60)
    t0 = time.time()
    export(scores)
    print(f"\nDone in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Cryostat RAG evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--all",          action="store_true", help="Run all steps end-to-end")
    parser.add_argument("--quick",        action="store_true",
                        help="Fast smoke test: subset of questions + variants, fewer bootstrap iters")
    parser.add_argument("--dataset",      action="store_true", help="Step 1: generate Q&A dataset")
    parser.add_argument("--pipeline",     action="store_true", help="Step 2: run RAG pipeline")
    parser.add_argument("--score",        action="store_true", help="Step 3: Vertex AI scoring")
    parser.add_argument("--export",       action="store_true", help="Step 4: export LaTeX + charts")
    parser.add_argument("--pdf-folder",   type=str, default=None,
                        help="Use local PDF folder instead of GCS (for --dataset)")
    parser.add_argument("--no-ablation",  action="store_true",
                        help="Skip ablation variants in --pipeline")
    parser.add_argument("--skip-rebuild", action="store_true",
                        help="Reuse existing local vectorstore in --pipeline")
    args = parser.parse_args()

    if not any([args.all, args.quick, args.dataset, args.pipeline, args.score, args.export]):
        parser.print_help()
        sys.exit(0)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # --quick alone runs pipeline -> score -> export on the existing dataset
    # (never regenerates the dataset; that is the expensive Gemini step).
    if args.quick and not any([args.dataset, args.pipeline, args.score, args.export]):
        args.pipeline = args.score = args.export = True

    dataset      = None
    rag_results  = None
    scores_data  = None

    if (args.all or args.dataset) and not args.quick:
        dataset = _step_dataset(args)

    if args.all or args.pipeline:
        rag_results = _step_pipeline(args, dataset=dataset)

    if args.all or args.score:
        scores_data = _step_score(args, rag_results=rag_results)

    if args.all or args.export:
        _step_export(args, scores=scores_data)

    if args.all:
        print("\n" + "=" * 60)
        print("Evaluation complete.")
        print(f"  Dataset:   {DATASET_PATH}")
        print(f"  Results:   {RAG_RESULTS_PATH}")
        print(f"  Scores:    {SCORES_PATH}")
        print(f"  Paper dir: eval/outputs/paper/")
        print("=" * 60)


if __name__ == "__main__":
    main()
