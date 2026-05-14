"""
Environment setup for running evaluation locally.

Must be imported BEFORE rag.py to set writable local paths and GCP config.
rag.py reads environment variables at module level (on import), so these
must be set before any `from rag import ...` statement executes.

Usage: Add `import evaluation.env_setup  # noqa: F401` as the first import
in any evaluation module that imports from rag.
"""
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_LOCAL_STORAGE = os.path.join(PROJECT_DIR, ".eval_data")

# Local storage paths (rag.py defaults to /var/lib/cryostat-rag which requires root)
if not os.environ.get("VECTORSTORE_BASE_DIR"):
    os.environ["VECTORSTORE_BASE_DIR"] = EVAL_LOCAL_STORAGE
if not os.environ.get("PDF_CACHE_DIR"):
    os.environ["PDF_CACHE_DIR"] = os.path.join(EVAL_LOCAL_STORAGE, "pdfs")

# GCP configuration (rag.py reads these at module level for vertexai.init())
if not os.environ.get("GCP_PROJECT_ID"):
    os.environ["GCP_PROJECT_ID"] = "mf-crucible"
if not os.environ.get("GCP_LOCATION"):
    os.environ["GCP_LOCATION"] = "us-central1"
if not os.environ.get("GCS_BUCKET_NAME"):
    os.environ["GCS_BUCKET_NAME"] = "attocube-rag-pdfs"
if not os.environ.get("VECTORSTORE_GCS_BUCKET"):
    os.environ["VECTORSTORE_GCS_BUCKET"] = "attocube-rag-vectordb"
