import os
import time
import json
import hashlib
import pickle
from datetime import datetime
from typing import Any, List, Tuple, Set, Dict, Optional

# Google Gen AI SDK (replaces the removed vertexai.generative_models /
# vertexai.language_models modules, retired June 24, 2026).
from google import genai
from google.genai import types
from google.cloud import storage
import fitz  # PyMuPDF for image detection

# community imports to avoid deprecation warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from google.cloud import discoveryengine_v1 as discoveryengine

from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# ─── 1. Initialize Vertex AI (Google Gen AI SDK, Vertex backend) ─────────────
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "mf-crucible")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# Generation uses gemini-3.6-flash and gemini-3.6-pro: the fastest current
# models, and higher-scoring than the retiring gemini-2.5-pro
# (Vertex retirement 2026-10-16). Embeddings stay on text-embedding-005 — it is
# NOT deprecated (only the old vertexai.language_models SDK module was removed),
# so keeping it avoids a vector-store rebuild and any embedding drift.
# All overridable via environment for easy A/B testing.
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-3.6-flash")
GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-3.6-pro")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "text-embedding-005")

# gemini-3.6-flash is currently served only from the global endpoint (regional
# endpoints like us-central1 return 404), while text-embedding-005 is regional.
# Hence two clients, one per endpoint.
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "global")
gen_client   = genai.Client(vertexai=True, project=PROJECT_ID, location=GEMINI_LOCATION)
embed_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# ─── 2. Chunking parameters ─────────────────────────────────────────────────
FINE_CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "600"))
FINE_CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "100"))
COARSE_CHUNK_SIZE    = int(os.getenv("COARSE_CHUNK_SIZE", "3000"))
COARSE_CHUNK_OVERLAP = int(os.getenv("COARSE_CHUNK_OVERLAP", "400"))
SEPARATORS           = ["\n\n", "\n", " ", ""]
CONTEXT_WINDOW_SIZE  = 3  # Number of previous queries to keep chunks for

# Context expansion parameters
EXPAND_CONTEXT_BEFORE = int(os.getenv("EXPAND_CONTEXT_BEFORE", "1"))
EXPAND_CONTEXT_AFTER = int(os.getenv("EXPAND_CONTEXT_AFTER", "2"))

# Vertex AI Ranking API reranking
USE_RERANKING     = os.getenv("USE_RERANKING", "true").lower() == "true"
RERANK_TOP_N      = int(os.getenv("RERANK_TOP_N", "8"))
RERANK_MODEL      = os.getenv("RERANK_MODEL", "semantic-ranker-fast-004")
_MAX_RECORD_CHARS = 3800  # ~1024-token safety truncation per API record
_MAX_RERANK_DOCS  = 200   # API hard limit per request
EMBED_BATCH_SIZE  = int(os.getenv("EMBED_BATCH_SIZE", "20"))

# ─── Ablation / configuration toggles ────────────────────────────────────────
# Production defaults preserve the full pipeline. The evaluation harness
# monkeypatches these per variant (see eval/pipeline.py) to isolate the
# contribution of each component; they are also overridable by environment var.
RETRIEVAL_MODE        = os.getenv("RETRIEVAL_MODE", "hybrid")  # hybrid | dense | bm25
USE_QUERY_ROUTING     = os.getenv("USE_QUERY_ROUTING", "true").lower() == "true"
USE_CONTEXT_EXPANSION = os.getenv("USE_CONTEXT_EXPANSION", "true").lower() == "true"
USE_DOMAIN_PROMPT     = os.getenv("USE_DOMAIN_PROMPT", "true").lower() == "true"

# GCS settings
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "attocube-rag-pdfs")
GCS_PDF_PREFIX = "pdfs/"

# Persistent storage paths (Cloud Run volume recommended)
VECTORSTORE_BASE_DIR = os.getenv("VECTORSTORE_BASE_DIR", "/var/lib/cryostat-rag")
VECTORSTORE_DIR = os.path.join(VECTORSTORE_BASE_DIR, "vectorstore")
PDF_CACHE_DIR = os.getenv("PDF_CACHE_DIR", os.path.join(VECTORSTORE_BASE_DIR, "pdfs"))
MANIFEST_PATH = os.path.join(VECTORSTORE_DIR, "manifest.json")
IMAGE_STORAGE_PATH = os.path.join(VECTORSTORE_DIR, "images.pkl")
BM25_DOCS_PATH = os.path.join(VECTORSTORE_DIR, "bm25_docs.pkl")
VECTORSTORE_GCS_BUCKET = os.getenv("VECTORSTORE_GCS_BUCKET")
VECTORSTORE_GCS_PREFIX = os.getenv("VECTORSTORE_GCS_PREFIX", "vectorstore/")

# Global image storage to avoid ChromaDB metadata issues
GLOBAL_IMAGE_STORAGE = {}

# Global BM25 document store for hybrid retrieval
GLOBAL_BM25_DOCS: List[Document] = []

# ─── 3. LLM wrapper for Vertex AI Gemini ────────────────────────────────────
class VertexAIGeminiLLM(LLM):
    model_name: str = GEMINI_FLASH_MODEL

    def __init__(self, model_name: str = None):
        super().__init__()
        if model_name:
            self.model_name = model_name

    @property
    def _llm_type(self) -> str:
        return "vertex-ai-gemini"

    def _call(self, prompt: str, stop=None) -> str:
        response = gen_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4096,
            ),
        )
        return response.text

# ─── 4. Embeddings wrapper for Vertex AI ─────────────────────────────────────
class VertexAIEmbeddings(Embeddings):
    def __init__(self, model_name: str = None):
        self.model_name = model_name or EMBED_MODEL

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # Plain (untyped) embeddings via the Gen AI SDK, matching the original
        # text-embedding-005 output exactly so the existing vector store stays valid.
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            for attempt in range(5):
                try:
                    resp = embed_client.models.embed_content(
                        model=self.model_name,
                        contents=batch,
                    )
                    embeddings.extend(list(e.values) for e in resp.embeddings)
                    break
                except Exception as e:
                    if "ResourceExhausted" in type(e).__name__ or "429" in str(e):
                        wait = (2 ** attempt) + 1
                        print(f"Embedding rate-limited (attempt {attempt+1}/5), retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            else:
                raise RuntimeError(f"Embedding batch {i // EMBED_BATCH_SIZE} failed after 5 retries")
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

# ─── 5. GCS PDF Loading ──────────────────────────────────────────────────────
def download_pdfs_from_gcs(bucket_name: str, prefix: str, local_dir: str = "pdfs"):
    """Download PDFs from GCS to local directory"""
    os.makedirs(local_dir, exist_ok=True)
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    
    # List and download all PDFs
    blobs = bucket.list_blobs(prefix=prefix)
    pdf_count = 0
    
    for blob in blobs:
        if blob.name.endswith('.pdf'):
            local_path = os.path.join(local_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            pdf_count += 1
            print(f"Downloaded: {blob.name}")
    
    print(f"Downloaded {pdf_count} PDFs from GCS")
    return local_dir

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _compute_sha256_for_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_manifest_from_gcs(bucket_name: str, prefix: str, status_callback=None) -> Dict[str, Any]:
    """Compute a change-detection manifest for PDFs in GCS.

    Uses the content fingerprint GCS already returns in the object listing
    (``md5_hash``, falling back to ``generation``) instead of streaming each
    file to hash it. A single list_blobs call is enough to detect changes, so a
    cold start no longer downloads every PDF just to fingerprint it.
    """
    if status_callback:
        status_callback("verifying", "Checking for document updates...")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith('.pdf')]

    files = []
    for blob in sorted(blobs, key=lambda b: b.name):
        # md5_hash is the base64 content hash from the listing metadata; if a
        # bucket disables it (e.g. composite objects), generation still changes
        # on every overwrite, so it is a sound fallback fingerprint.
        fingerprint = blob.md5_hash or (str(blob.generation) if blob.generation else None)
        files.append({
            "name": os.path.basename(blob.name),
            "md5": fingerprint,
            "size": blob.size,
            "updated": blob.updated.isoformat() if blob.updated else None
        })

    manifest = {
        "bucket": bucket_name,
        "prefix": prefix,
        "hash_algo": "md5",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "files": files
    }
    manifest["manifest_hash"] = _manifest_hash(manifest)
    return manifest

def compute_manifest_from_local(folder: str) -> Dict[str, Any]:
    files = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, fname)
        files.append({
            "name": fname,
            "sha256": _compute_sha256_for_file(path),
            "size": os.path.getsize(path),
            "updated": datetime.utcfromtimestamp(os.path.getmtime(path)).isoformat() + "Z"
        })
    manifest = {
        "bucket": None,
        "prefix": None,
        "hash_algo": "sha256",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "files": files
    }
    manifest["manifest_hash"] = _manifest_hash(manifest)
    return manifest

def _manifest_hash(manifest: Dict[str, Any]) -> str:
    compact = json.dumps({
        "bucket": manifest.get("bucket"),
        "prefix": manifest.get("prefix"),
        "hash_algo": manifest.get("hash_algo"),
        "files": manifest.get("files", [])
    }, sort_keys=True)
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()

def load_manifest() -> Optional[Dict[str, Any]]:
    if not os.path.exists(MANIFEST_PATH):
        return None
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_manifest(manifest: Dict[str, Any]):
    _ensure_dir(VECTORSTORE_DIR)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

def manifest_has_changed(old: Optional[Dict[str, Any]], new: Dict[str, Any]) -> bool:
    if not old:
        return True
    return old.get("manifest_hash") != new.get("manifest_hash")

def save_image_storage():
    _ensure_dir(VECTORSTORE_DIR)
    with open(IMAGE_STORAGE_PATH, "wb") as f:
        pickle.dump(GLOBAL_IMAGE_STORAGE, f)

def load_image_storage():
    if not os.path.exists(IMAGE_STORAGE_PATH):
        return
    with open(IMAGE_STORAGE_PATH, "rb") as f:
        data = pickle.load(f)
        GLOBAL_IMAGE_STORAGE.clear()
        GLOBAL_IMAGE_STORAGE.update(data)

def save_bm25_docs(docs: List[Document]):
    global GLOBAL_BM25_DOCS
    _ensure_dir(VECTORSTORE_DIR)
    with open(BM25_DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)
    GLOBAL_BM25_DOCS = docs  # Make available immediately for retriever initialization
    print(f"Saved {len(docs)} docs for BM25 index.")

def load_bm25_docs():
    global GLOBAL_BM25_DOCS
    if not os.path.exists(BM25_DOCS_PATH):
        return
    with open(BM25_DOCS_PATH, "rb") as f:
        GLOBAL_BM25_DOCS = pickle.load(f)
    print(f"Loaded {len(GLOBAL_BM25_DOCS)} docs for BM25 index.")

def _download_prefix(bucket_name: str, prefix: str, local_dir: str):
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    _ensure_dir(local_dir)
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith('/'):
            continue
        rel_path = blob.name[len(prefix):]
        local_path = os.path.join(local_dir, rel_path)
        _ensure_dir(os.path.dirname(local_path))
        blob.download_to_filename(local_path)

def _upload_dir(bucket_name: str, prefix: str, local_dir: str):
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir).replace("\\", "/")
            blob = bucket.blob(prefix + rel_path)
            blob.upload_from_filename(local_path)

def sync_vectorstore_from_gcs():
    if VECTORSTORE_GCS_BUCKET:
        _download_prefix(VECTORSTORE_GCS_BUCKET, VECTORSTORE_GCS_PREFIX, VECTORSTORE_DIR)

def sync_vectorstore_to_gcs():
    if VECTORSTORE_GCS_BUCKET:
        _upload_dir(VECTORSTORE_GCS_BUCKET, VECTORSTORE_GCS_PREFIX, VECTORSTORE_DIR)

# Keep all the original chunking and retriever logic...

def load_and_split_pdfs(folder: str) -> Tuple[List[Document], List[Document]]:
    fine_splitter = RecursiveCharacterTextSplitter(
        chunk_size=FINE_CHUNK_SIZE,
        chunk_overlap=FINE_CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    coarse_splitter = RecursiveCharacterTextSplitter(
        chunk_size=COARSE_CHUNK_SIZE,
        chunk_overlap=COARSE_CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    fine_docs, coarse_docs = [], []

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".pdf"): continue
        path = os.path.join(folder, fname)

        # Detect and extract image information
        doc_pdf = fitz.open(path)
        pages_with_images = {}
        for i, pg in enumerate(doc_pdf):
            page_num = i + 1
            images = pg.get_images()
            if images:
                pages_with_images[page_num] = []
                for img_index, img in enumerate(images):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(doc_pdf, xref)
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            pages_with_images[page_num].append({
                                "image_index": img_index,
                                "image_data": img_data,
                                "width": pix.width,
                                "height": pix.height
                            })
                        if pix:
                            pix = None
                    except Exception as e:
                        print(f"Failed to extract image {img_index} from page {page_num} of {fname}: {e}")
                        continue
        doc_pdf.close()
        
        # Print image extraction summary
        total_images = sum(len(imgs) for imgs in pages_with_images.values())
        if total_images > 0:
            print(f"Extracted {total_images} images from {fname} across {len(pages_with_images)} pages")

        # Classify document type based on filename
        doc_type = "email" if fname.startswith("UHD") else "manual"
        
        pages = PyPDFLoader(path).load()
        for i, doc in enumerate(pages, start=1):
            doc.metadata.update({
                "source": fname,
                "page": i,
                "has_image": i in pages_with_images,
                "doc_type": doc_type,
                "image_count": len(pages_with_images.get(i, []))
            })
            # Store images separately to avoid ChromaDB metadata issues
            if i in pages_with_images:
                image_key = f"{fname}_page_{i}"
                GLOBAL_IMAGE_STORAGE[image_key] = pages_with_images[i]
        
        # Split and add chunk indices
        fine_chunks = fine_splitter.split_documents(pages)
        coarse_chunks = coarse_splitter.split_documents(pages)
        
        # Add chunk index within the document
        for idx, chunk in enumerate(fine_chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = len(fine_chunks)
            chunk.metadata["doc_id"] = fname
        
        for idx, chunk in enumerate(coarse_chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = len(coarse_chunks)
            chunk.metadata["doc_id"] = fname
        
        fine_docs.extend(fine_chunks)
        coarse_docs.extend(coarse_chunks)

    return fine_docs, coarse_docs

# ─── 6. ChromaDB setup ───────────────────────────────────────────────────────
CHROMA_BASE_DIR   = os.path.join(VECTORSTORE_DIR, "chroma")
CHROMA_FINE_DIR   = os.path.join(CHROMA_BASE_DIR, "fine")
CHROMA_COARSE_DIR = os.path.join(CHROMA_BASE_DIR, "coarse")

def get_vectorstores(fine_docs: List[Document], coarse_docs: List[Document]):
    os.makedirs(CHROMA_FINE_DIR, exist_ok=True)
    os.makedirs(CHROMA_COARSE_DIR, exist_ok=True)
    emb = VertexAIEmbeddings()
    
    print("Building vector stores...")
    fine_db = Chroma.from_documents(fine_docs, emb, persist_directory=CHROMA_FINE_DIR)
    coarse_db = Chroma.from_documents(coarse_docs, emb, persist_directory=CHROMA_COARSE_DIR)
    print("Vector stores built successfully!")
    
    return fine_db, coarse_db

def _vectorstore_exists() -> bool:
    fine_db_file = os.path.join(CHROMA_FINE_DIR, "chroma.sqlite3")
    coarse_db_file = os.path.join(CHROMA_COARSE_DIR, "chroma.sqlite3")
    return os.path.exists(fine_db_file) and os.path.exists(coarse_db_file)

def load_vectorstores_if_present() -> Tuple[Optional[Chroma], Optional[Chroma], bool]:
    if not _vectorstore_exists():
        return None, None, False
    emb = VertexAIEmbeddings()
    fine_db = Chroma(persist_directory=CHROMA_FINE_DIR, embedding_function=emb)
    coarse_db = Chroma(persist_directory=CHROMA_COARSE_DIR, embedding_function=emb)
    load_image_storage()
    load_bm25_docs()
    return fine_db, coarse_db, True

def _clear_directory(folder: str):
    if not os.path.exists(folder):
        return
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            os.remove(path)

def build_vectorstores_from_gcs(status_callback=None, manifest: Optional[Dict[str, Any]] = None) -> Tuple[Chroma, Chroma, Dict[str, Any]]:
    if status_callback:
        status_callback("downloading", "Downloading PDFs from storage...")
    _ensure_dir(PDF_CACHE_DIR)
    _clear_directory(PDF_CACHE_DIR)
    pdf_folder = download_pdfs_from_gcs(GCS_BUCKET_NAME, GCS_PDF_PREFIX, PDF_CACHE_DIR)

    if status_callback:
        status_callback("chunking", "Parsing and chunking documents...")
    fine_docs, coarse_docs = load_and_split_pdfs(pdf_folder)

    if status_callback:
        status_callback("embedding", "Creating embeddings and vector store...")
    fine_db, coarse_db = get_vectorstores(fine_docs, coarse_docs)

    save_image_storage()
    save_bm25_docs(fine_docs)
    # Reuse the manifest computed during verification when available; only fall
    # back to computing it here (one metadata listing, no downloads).
    if manifest is None:
        manifest = compute_manifest_from_gcs(GCS_BUCKET_NAME, GCS_PDF_PREFIX, status_callback=status_callback)
    save_manifest(manifest)
    sync_vectorstore_to_gcs()
    return fine_db, coarse_db, manifest

def verify_or_rebuild_vectorstores(status_callback=None) -> Tuple[Chroma, Chroma, Dict[str, Any], bool]:
    """Verify GCS manifest and rebuild vectorstores if changed or missing."""
    if status_callback:
        status_callback("verifying", "Checking for document updates...")

    existing_manifest = load_manifest()
    new_manifest = compute_manifest_from_gcs(GCS_BUCKET_NAME, GCS_PDF_PREFIX, status_callback=status_callback)
    needs_rebuild = manifest_has_changed(existing_manifest, new_manifest) or not _vectorstore_exists()

    if needs_rebuild:
        if status_callback:
            status_callback("rebuilding", "Rebuilding knowledge base...")
        fine_db, coarse_db, manifest = build_vectorstores_from_gcs(
            status_callback=status_callback, manifest=new_manifest
        )
        return fine_db, coarse_db, manifest, True

    fine_db, coarse_db, _ = load_vectorstores_if_present()
    return fine_db, coarse_db, existing_manifest or new_manifest, False

# Include all your original retriever and conversation classes here...
# (ContextExpandingHybridRetriever, ConversationHistory, etc.)
# I'll skip them for brevity but they remain the same

class ContextExpandingHybridRetriever:
    """Hybrid retriever that expands context by including neighboring chunks"""
    def __init__(self, fine_db, coarse_db):
        self.fine_db = fine_db
        self.coarse_db = coarse_db
        self.fine_retriever = fine_db.as_retriever(search_kwargs={"k": 12})
        self.coarse_retriever = coarse_db.as_retriever(search_kwargs={"k": 6})
        # BM25 index for hybrid retrieval
        if GLOBAL_BM25_DOCS:
            self.bm25_retriever = BM25Retriever.from_documents(GLOBAL_BM25_DOCS, k=12)
            print(f"BM25 retriever initialized with {len(GLOBAL_BM25_DOCS)} documents.")
        else:
            self.bm25_retriever = None
            print("BM25 retriever not initialized (no docs in GLOBAL_BM25_DOCS).")

    def _reciprocal_rank_fusion(self, results_lists: List[List[Document]], k: int = 60) -> List[Document]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
        scores: Dict[str, Dict] = {}
        for results in results_lists:
            for rank, doc in enumerate(results):
                key = doc.page_content[:120]
                if key not in scores:
                    scores[key] = {"score": 0.0, "doc": doc}
                scores[key]["score"] += 1.0 / (k + rank + 1)
        sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_items]

    def _is_procedural(self, query: str) -> bool:
        """Keyword/prefix routing to the coarse index. Disabled when USE_QUERY_ROUTING is off."""
        if not USE_QUERY_ROUTING:
            return False
        lower_q = query.lower()
        _HOW_ACTION = ("how to ", "how do ", "how does ", "how did ",
                       "how should ", "how would ", "how can ", "how could ", "how must ")
        return (
            any(k in lower_q for k in ["procedure", "steps", "process", "install", "setup"])
            or any(lower_q.startswith(p) or f" {p}" in lower_q for p in _HOW_ACTION)
        )

    def _fine_fuse(self, query: str) -> List[Document]:
        """Fine-index retrieval honoring RETRIEVAL_MODE (hybrid | dense | bm25)."""
        dense_docs = (self.fine_retriever.get_relevant_documents(query)
                      if RETRIEVAL_MODE in ("hybrid", "dense") else [])
        bm25_docs = (self.bm25_retriever.get_relevant_documents(query)
                     if self.bm25_retriever and RETRIEVAL_MODE in ("hybrid", "bm25") else [])
        if RETRIEVAL_MODE == "dense":
            return dense_docs
        if RETRIEVAL_MODE == "bm25":
            return bm25_docs
        if bm25_docs:
            return self._reciprocal_rank_fusion([dense_docs, bm25_docs])
        return dense_docs

    def rerank(self, query: str, docs: List[Document], top_n: int = None) -> List[Document]:
        """Rerank docs via Vertex AI Ranking API; falls back gracefully on error."""
        if not USE_RERANKING or not docs:
            return docs
        if top_n is None:
            top_n = RERANK_TOP_N
        if len(docs) <= top_n:
            return docs

        candidates = docs[:_MAX_RERANK_DOCS]
        client = discoveryengine.RankServiceClient()
        ranking_config = (
            f"projects/{PROJECT_ID}/locations/global"
            "/rankingConfigs/default_ranking_config"
        )
        records = [
            discoveryengine.RankingRecord(
                id=str(i),
                content=doc.page_content[:_MAX_RECORD_CHARS],
            )
            for i, doc in enumerate(candidates)
        ]
        try:
            response = client.rank(
                discoveryengine.RankRequest(
                    ranking_config=ranking_config,
                    model=RERANK_MODEL,
                    query=query,
                    records=records,
                    top_n=top_n,
                )
            )
            id_map = {str(i): doc for i, doc in enumerate(candidates)}
            return [id_map[r.id] for r in response.records if r.id in id_map]
        except Exception as e:
            print(f"Reranking API error (falling back to unranked top-{top_n}): {e}")
            return docs[:top_n]
    
    def expand_context(self, docs: List[Document], db: Chroma, before: int = 1, after: int = 2) -> List[Document]:
        """Expand context by including neighboring chunks"""
        if not USE_CONTEXT_EXPANSION:
            return docs
        expanded_docs = []
        seen_chunks = set()
        
        for doc in docs:
            doc_id = doc.metadata.get("doc_id")
            chunk_idx = doc.metadata.get("chunk_index")
            
            if doc_id is None or chunk_idx is None:
                expanded_docs.append(doc)
                continue
            
            # Add chunks before
            for i in range(before, 0, -1):
                target_idx = chunk_idx - i
                if target_idx >= 0:
                    neighbor = self._get_chunk_by_index(db, doc_id, target_idx)
                    if neighbor and (doc_id, target_idx) not in seen_chunks:
                        expanded_docs.append(neighbor)
                        seen_chunks.add((doc_id, target_idx))
            
            # Add the original chunk
            if (doc_id, chunk_idx) not in seen_chunks:
                expanded_docs.append(doc)
                seen_chunks.add((doc_id, chunk_idx))
            
            # Add chunks after
            total_chunks = doc.metadata.get("total_chunks", float('inf'))
            for i in range(1, after + 1):
                target_idx = chunk_idx + i
                if target_idx < total_chunks:
                    neighbor = self._get_chunk_by_index(db, doc_id, target_idx)
                    if neighbor and (doc_id, target_idx) not in seen_chunks:
                        expanded_docs.append(neighbor)
                        seen_chunks.add((doc_id, target_idx))
        
        return expanded_docs
    
    def _get_chunk_by_index(self, db: Chroma, doc_id: str, chunk_index: int) -> Optional[Document]:
        """Retrieve a specific chunk by document ID and chunk index"""
        results = db.get(
            where={"$and": [
                {"doc_id": {"$eq": doc_id}},
                {"chunk_index": {"$eq": chunk_index}}
            ]},
            limit=1
        )
        
        if results and results['documents']:
            return Document(
                page_content=results['documents'][0],
                metadata=results['metadatas'][0]
            )
        return None
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Main retrieval method with BM25+dense hybrid retrieval and context expansion.

        BM25 hybrid is applied only for factual queries (non-procedural).
        Procedural queries use coarse-only dense retrieval since BM25 is indexed
        from fine chunks and cannot be correctly expanded against the coarse DB.
        """
        if self._is_procedural(query):
            # Coarse dense retrieval only — BM25 (fine-indexed) would mismatch coarse expansion
            docs = self.coarse_retriever.get_relevant_documents(query)
            expanded = self.expand_context(
                docs,
                self.coarse_db,
                before=EXPAND_CONTEXT_BEFORE,
                after=EXPAND_CONTEXT_AFTER + 1
            )
        else:
            # Factual: BM25+dense hybrid over fine DB
            fused = self._fine_fuse(query)
            expanded = self.expand_context(
                fused,
                self.fine_db,
                before=EXPAND_CONTEXT_BEFORE,
                after=EXPAND_CONTEXT_AFTER
            )

        return expanded

    def get_relevant_documents_by_type(self, query: str, doc_type: str = None) -> List[Document]:
        """Retrieval method with optional document type filtering"""
        if self._is_procedural(query):
            if doc_type:
                # Use filtered retriever for specific document types
                docs = self._get_filtered_documents(query, self.coarse_db, doc_type, k=2)
            else:
                docs = self.coarse_retriever.get_relevant_documents(query)
            expanded = self.expand_context(
                docs, 
                self.coarse_db, 
                before=EXPAND_CONTEXT_BEFORE, 
                after=EXPAND_CONTEXT_AFTER + 1
            )
        else:
            if doc_type:
                # Use filtered retriever for specific document types
                docs = self._get_filtered_documents(query, self.fine_db, doc_type, k=3)
            else:
                docs = self.fine_retriever.get_relevant_documents(query)
            expanded = self.expand_context(
                docs, 
                self.fine_db, 
                before=EXPAND_CONTEXT_BEFORE, 
                after=EXPAND_CONTEXT_AFTER
            )
        
        return expanded
    
    def get_relevant_documents_with_expansion(self, query: str, before: int = None, after: int = None) -> List[Document]:
        """Retrieval with custom context expansion. BM25 hybrid applied for factual queries only."""
        if before is None:
            before = EXPAND_CONTEXT_BEFORE
        if after is None:
            after = EXPAND_CONTEXT_AFTER

        if self._is_procedural(query):
            docs = self.coarse_retriever.get_relevant_documents(query)
            expanded = self.expand_context(docs, self.coarse_db, before=before, after=after + 1)
        else:
            fused = self._fine_fuse(query)
            expanded = self.expand_context(fused, self.fine_db, before=before, after=after)

        return expanded
    
    def get_relevant_documents_by_type_with_expansion(self, query: str, doc_type: str = None, before: int = None, after: int = None) -> List[Document]:
        """Retrieval method with optional document type filtering and custom context expansion"""
        if before is None:
            before = EXPAND_CONTEXT_BEFORE
        if after is None:
            after = EXPAND_CONTEXT_AFTER

        if self._is_procedural(query):
            if doc_type:
                docs = self._get_filtered_documents(query, self.coarse_db, doc_type, k=2)
            else:
                docs = self.coarse_retriever.get_relevant_documents(query)
            expanded = self.expand_context(docs, self.coarse_db, before=before, after=after + 1)
        else:
            if doc_type:
                docs = self._get_filtered_documents(query, self.fine_db, doc_type, k=3)
            else:
                docs = self.fine_retriever.get_relevant_documents(query)
            expanded = self.expand_context(docs, self.fine_db, before=before, after=after)
        
        return expanded
    
    def _get_filtered_documents(self, query: str, db: Chroma, doc_type: str, k: int = 3) -> List[Document]:
        """Get documents filtered by document type"""
        # Perform similarity search with metadata filter
        docs = db.similarity_search(
            query, 
            k=k*3,  # Get more docs initially to account for filtering
            filter={"doc_type": doc_type}
        )
        # Return top k after filtering
        return docs[:k]
    
    def invoke(self, input: str, config=None) -> List[Document]:
        return self.get_relevant_documents(input)
    
    async def ainvoke(self, input: str, config=None) -> List[Document]:
        return self.get_relevant_documents(input)

class ConversationHistory:
    def __init__(self):
        self.messages = []
        self.awaiting_clarification = False
        self.original_question = None
        self.chunk_history = []
        self.max_chunk_history = CONTEXT_WINDOW_SIZE
    
    def add_user_message(self, message: str):
        self.messages.append(HumanMessage(content=message))
    
    def add_ai_message(self, message: str):
        self.messages.append(AIMessage(content=message))
    
    def add_chunks_to_history(self, question: str, chunks: List[Document]):
        self.chunk_history.append((question, chunks))
        if len(self.chunk_history) > self.max_chunk_history:
            self.chunk_history.pop(0)
    
    def get_recent_chunks(self, n: int = 2) -> List[Document]:
        all_chunks = []
        for _, chunks in self.chunk_history[-n:]:
            all_chunks.extend(chunks)
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk.page_content not in seen:
                seen.add(chunk.page_content)
                unique_chunks.append(chunk)
        return unique_chunks
    
    def get_messages(self):
        return self.messages
    
    def set_clarification_mode(self, question: str):
        self.awaiting_clarification = True
        self.original_question = question
    
    def clear_clarification_mode(self):
        self.awaiting_clarification = False
        self.original_question = None
    
    def get_conversation_context(self, n: int = 3) -> str:
        recent = self.messages[-n*2:] if len(self.messages) >= n*2 else self.messages
        context = []
        for msg in recent:
            if isinstance(msg, HumanMessage):
                context.append(f"User: {msg.content}")
            else:
                context.append(f"Assistant: {msg.content[:200]}...")
        return "\n".join(context)
    
    def clear(self):
        self.messages = []
        self.awaiting_clarification = False
        self.original_question = None
        self.chunk_history = []

def is_follow_up_query(query: str) -> bool:
    lower_q = query.lower().strip()
    words = lower_q.split()

    # Domain-specific terms that strongly indicate a standalone question
    _DOMAIN_TERMS = {
        "attodry", "anc300", "anc", "uhd", "flex", "magnet", "compressor",
        "temperature", "pressure", "voltage", "current", "sample", "stage",
        "positioner", "controller", "valve", "cryostat", "helium", "nitrogen",
        "cooldown", "warmup", "vacuum", "sensor", "heater", "pid",
    }

    # Explicit continuation phrases — high confidence follow-up regardless of length
    _EXPLICIT = [
        "tell me more", "what else", "anything else", "more details",
        "explain that", "explain this", "elaborate", "go on", "continue",
        "can you elaborate", "can you clarify", "what did you mean",
        "clarify that", "clarify this", "expand on that", "expand on this",
        "what does that mean", "what does this mean",
    ]
    if any(phrase in lower_q for phrase in _EXPLICIT):
        return True

    # Very short query (≤3 words) with no domain term → likely contextual
    if len(words) <= 3 and not any(t in lower_q for t in _DOMAIN_TERMS):
        return True

    # Short queries (≤8 words) that open with or contain an unresolved pronoun
    # and carry no domain term to self-resolve the reference
    _PRONOUNS = ("it ", "that ", "this ", "they ", "them ", "those ", "its ")
    if len(words) <= 8 and not any(t in lower_q for t in _DOMAIN_TERMS):
        if any(lower_q.startswith(p) or f" {p}" in lower_q for p in _PRONOUNS):
            return True

    return False

def needs_clarification(query: str) -> bool:
    vague_queries = [
        "what", "how", "why", "when", "where", "who",
        "tell me", "explain", "help", "info", "information",
        "details", "specs", "?"
    ]
    
    query_lower = query.lower().strip()
    words = query_lower.split()
    
    if len(words) <= 2 and any(vague == query_lower for vague in vague_queries):
        return True
    
    if any(char.isdigit() for char in query):
        return False
    if len(words) > 5:
        return False
    if any(word in query_lower for word in ["model", "serial", "temperature", "pressure", "voltage", "current", "power", "size", "dimension"]):
        return False
    
    return False

def extract_images_from_chunks(docs: List[Document]) -> List[Dict]:
    """Extract images from document chunks that contain images"""
    images = []
    seen_images = set()
    
    for doc in docs:
        # Check if this chunk has images using metadata
        if doc.metadata.get("has_image", False):
            source = doc.metadata.get('source', '')
            page = doc.metadata.get('page', 0)
            
            # Get image key for this page
            image_key = f"{source}_page_{page}"
            
            # Check if this page has images in global storage
            if image_key in GLOBAL_IMAGE_STORAGE:
                page_images = GLOBAL_IMAGE_STORAGE[image_key]
                for img_info in page_images:
                    # Create unique identifier for image
                    img_id = f"{source}_page{page}_img{img_info['image_index']}"
                    
                    if img_id not in seen_images:
                        images.append({
                            "id": img_id,
                            "source": source,
                            "page": page,
                            "doc_type": doc.metadata.get("doc_type"),
                            "image_data": img_info["image_data"],
                            "width": img_info["width"],
                            "height": img_info["height"],
                            "image_index": img_info["image_index"]
                        })
                        seen_images.add(img_id)
    
    return images

# ─── 7. Initialize RAG system ────────────────────────────────────────────────
def initialize_rag_system(status_callback=None):
    """Initialize the RAG system by rebuilding vector stores from GCS."""
    print("Initializing RAG system (full rebuild)...")
    fine_db, coarse_db, _manifest = build_vectorstores_from_gcs(status_callback=status_callback)
    retriever = ContextExpandingHybridRetriever(fine_db, coarse_db)
    flash_llm = VertexAIGeminiLLM(GEMINI_FLASH_MODEL)
    pro_llm = VertexAIGeminiLLM(GEMINI_PRO_MODEL)
    print("RAG system initialized successfully!")
    return retriever, (flash_llm, pro_llm)

def load_rag_if_available() -> Tuple[Optional[ContextExpandingHybridRetriever], Optional[Tuple[VertexAIGeminiLLM, VertexAIGeminiLLM]], bool]:
    fine_db, coarse_db, loaded = load_vectorstores_if_present()
    if not loaded:
        return None, None, False
    retriever = ContextExpandingHybridRetriever(fine_db, coarse_db)
    flash_llm = VertexAIGeminiLLM(GEMINI_FLASH_MODEL)
    pro_llm = VertexAIGeminiLLM(GEMINI_PRO_MODEL)
    return retriever, (flash_llm, pro_llm), True

def verify_or_rebuild_rag(status_callback=None) -> Tuple[ContextExpandingHybridRetriever, Tuple[VertexAIGeminiLLM, VertexAIGeminiLLM], Dict[str, Any]]:
    fine_db, coarse_db, manifest, rebuilt = verify_or_rebuild_vectorstores(status_callback=status_callback)
    retriever = ContextExpandingHybridRetriever(fine_db, coarse_db)
    flash_llm = VertexAIGeminiLLM(GEMINI_FLASH_MODEL)
    pro_llm = VertexAIGeminiLLM(GEMINI_PRO_MODEL)
    return retriever, (flash_llm, pro_llm), {
        "rebuilt": rebuilt,
        "manifest": manifest
    }

# ─── 8. Query processing function for API ────────────────────────────────────
def process_query(query: str, retriever, llms, conversation_history, debug_mode: bool = False, status_callback=None):
    """Process a single query and return response with optional debug info"""
    flash_llm, pro_llm = llms

    # Vagueness / clarification branch.
    # A query is treated as a contextual follow-up only when there is prior history to
    # resolve it against; otherwise a one-word query like "what" has nothing to follow up.
    is_contextual_followup = is_follow_up_query(query) and len(conversation_history.messages) > 0

    if conversation_history.awaiting_clarification:
        # This turn is the user's reply to a clarifying question — resume normal retrieval.
        conversation_history.clear_clarification_mode()
    elif needs_clarification(query) and not is_contextual_followup:
        # Too vague to retrieve against, and not a follow-up: ask once instead of guessing.
        clarifying = (
            "Could you give me a bit more detail so I can find the right information? "
            "For example, name the instrument or component you're asking about "
            "(e.g., attoDRY2100, ANC300, magnet power supply) and what you'd like to know."
        )
        conversation_history.set_clarification_mode(query)
        conversation_history.add_user_message(query)
        conversation_history.add_ai_message(clarifying)
        return {
            "answer": clarifying,
            "sources": [],
            "images": [],
            "debug_info": None,
            "needs_clarification": True,
        }

    if status_callback:
        status_callback("searching", "Searching knowledge base...")
    
    # Query reformulation logic
    if is_follow_up_query(query) and len(conversation_history.messages) > 0:
        if status_callback:
            status_callback("reformulating", "Understanding context...")
            
        reformulation_prompt = PromptTemplate.from_template(
            """Given the conversation history and a follow-up question, reformulate the question to be self-contained and specific.

Conversation history:
{history}

Follow-up question: {question}

Reformulated question (be specific and include context from the conversation):"""
        )
        reformulation_chain = reformulation_prompt | flash_llm | StrOutputParser()
        
        history_context = conversation_history.get_conversation_context()
        reformulated_q = reformulation_chain.invoke({
            "history": history_context,
            "question": query
        })
        effective_query = reformulated_q
    else:
        effective_query = query
    
    # Detect if user is asking for specific document type
    def detect_document_type(query_text: str) -> str:
        """Detect if user is asking for emails or manuals specifically"""
        lower_q = query_text.lower()
        
        email_keywords = ["email", "emails", "uhd", "correspondence", "message", "communication"]
        manual_keywords = ["manual", "manuals", "documentation", "guide", "handbook", "instruction"]
        
        email_score = sum(1 for keyword in email_keywords if keyword in lower_q)
        manual_score = sum(1 for keyword in manual_keywords if keyword in lower_q)
        
        if email_score > 0 and email_score > manual_score:
            return "email"
        elif manual_score > 0 and manual_score > email_score:
            return "manual"
        else:
            return None
    
    # Check for document type filtering
    doc_type_filter = detect_document_type(effective_query)
    
    # Determine if this is a follow-up query for context expansion
    is_followup = is_follow_up_query(query)
    
    if status_callback:
        if is_followup:
            status_callback("expanding", "Expanding context...")
        else:
            status_callback("searching", "Searching knowledge base...")
    
    # Get documents (with optional filtering and expanded context for follow-ups)
    if doc_type_filter:
        if is_followup:
            # Temporarily increase context expansion for follow-up queries
            original_before = retriever.fine_db._expand_before if hasattr(retriever.fine_db, '_expand_before') else EXPAND_CONTEXT_BEFORE
            original_after = retriever.fine_db._expand_after if hasattr(retriever.fine_db, '_expand_after') else EXPAND_CONTEXT_AFTER
            
            # Get documents with expanded context
            docs = retriever.get_relevant_documents_by_type_with_expansion(
                effective_query, doc_type_filter, 
                before=EXPAND_CONTEXT_BEFORE + 1, 
                after=EXPAND_CONTEXT_AFTER + 1
            )
        else:
            docs = retriever.get_relevant_documents_by_type(effective_query, doc_type_filter)
    else:
        if is_followup:
            # Get documents with expanded context for follow-up queries
            docs = retriever.get_relevant_documents_with_expansion(
                effective_query,
                before=EXPAND_CONTEXT_BEFORE + 1,
                after=EXPAND_CONTEXT_AFTER + 1
            )
        else:
            docs = retriever.get_relevant_documents(effective_query)
    
    print(f"DEBUG: Query: {effective_query}")
    print(f"DEBUG: Is follow-up query: {is_followup}")
    print(f"DEBUG: Doc type filter: {doc_type_filter}")
    print(f"DEBUG: Context expansion - Before: {EXPAND_CONTEXT_BEFORE + (1 if is_followup else 0)}, After: {EXPAND_CONTEXT_AFTER + (1 if is_followup else 0)}")
    print(f"DEBUG: Retrieved {len(docs)} documents before reranking")

    # Vertex AI reranking: score all candidates and keep top-N most relevant
    if hasattr(retriever, "rerank"):
        if status_callback:
            status_callback("reranking", "Reranking results...")
        docs = retriever.rerank(effective_query, docs)
        print(f"DEBUG: {len(docs)} documents after reranking")

    for i, doc in enumerate(docs[:3]):  # Show first 3 docs
        print(f"DEBUG: Doc {i+1} - Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}")
        print(f"DEBUG: Doc {i+1} - Content preview: {doc.page_content[:100]}...")

    conversation_history.add_chunks_to_history(effective_query, docs)
    
    # Get previous chunks if this is a follow-up query
    if is_followup:
        previous_chunks = conversation_history.get_recent_chunks(n=2)
    else:
        previous_chunks = []
    
    # Extract images from retrieved chunks (for separate output, not LLM)
    all_chunks_for_images = docs[:]
    if previous_chunks:
        all_chunks_for_images.extend(previous_chunks)
    images = extract_images_from_chunks(all_chunks_for_images)
    
    # Prepare debug info
    debug_info = None
    if debug_mode:
        debug_info = {
            "current_chunks": [],
            "previous_chunks": [],
            "doc_type_filter": doc_type_filter,
            "images_found": len(images)
        }
        
        for i, d in enumerate(docs, 1):
            debug_info["current_chunks"].append({
                "index": i,
                "chunk_index": d.metadata.get("chunk_index"),
                "total_chunks": d.metadata.get("total_chunks"),
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "doc_type": d.metadata.get("doc_type"),
                "has_image": d.metadata.get("has_image", False),
                "image_count": d.metadata.get("image_count", 0),
                "preview": d.page_content[:200].replace('\n', ' ')
            })
        
        for i, d in enumerate(previous_chunks, 1):
            debug_info["previous_chunks"].append({
                "index": i,
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "doc_type": d.metadata.get("doc_type"),
                "has_image": d.metadata.get("has_image", False),
                "image_count": d.metadata.get("image_count", 0),
                "preview": d.page_content[:100].replace('\n', ' ')
            })
    
    # ── Citation map ──────────────────────────────────────────────────────────
    # Assign a stable citation number to each unique (source, page) among the
    # retrieved docs. The same numbers are (a) embedded in the context the LLM
    # sees so it can cite them inline as [n], and (b) returned as `citations` so
    # the frontend can turn each [n] into a clickable reference.
    citation_map: Dict[Tuple[Optional[str], Optional[int]], int] = {}
    citations: List[Dict[str, Any]] = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key not in citation_map:
            number = len(citations) + 1
            citation_map[key] = number
            citations.append({
                "number": number,
                "filename": d.metadata.get("source"),
                "doc_type": d.metadata.get("doc_type"),
                "page": d.metadata.get("page"),
            })

    def format_docs_with_citations(docs):
        if not docs:
            return "No relevant context found."
        blocks = []
        for doc in docs:
            n = citation_map[(doc.metadata.get("source"), doc.metadata.get("page"))]
            src = doc.metadata.get("source")
            page = doc.metadata.get("page")
            blocks.append(f"[{n}] (Source: {src}, page {page})\n{doc.page_content}")
        return "\n\n---\n\n".join(blocks)

    def format_docs(docs):
        if not docs:
            return "No relevant context found."
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Format the context strings. The current context carries [n] labels for
    # inline citation; previous context is supplementary and stays unlabeled.
    context_str = format_docs_with_citations(docs)
    previous_context_str = format_docs(previous_chunks) if previous_chunks else "None"
    
    # Debug logging to check context
    print(f"DEBUG: Retrieved {len(docs)} documents")
    print(f"DEBUG: Context length: {len(context_str)} characters")
    print(f"DEBUG: Context preview: {context_str[:200]}...")
    print(f"DEBUG: Previous context length: {len(previous_context_str)} characters")
    
    # Additional debug: Check if context is actually populated
    if not docs:
        print("WARNING: No documents retrieved for query!")
    if context_str == "No relevant context found.":
        print("WARNING: Context string indicates no relevant context found!")
    else:
        print(f"DEBUG: Context contains data from {len(docs)} documents")
    
    # Create prompt template without f-string formatting to preserve template variables.
    # The generic prompt is the ablation baseline (USE_DOMAIN_PROMPT=False); production
    # uses the domain-specific prompt below.
    # Each chunk in Current Context is prefixed with a citation marker like
    # "[2] (Source: ..., page N)". The model is asked to reuse those numbers as
    # inline citations so answers are traceable to specific source chunks.
    citation_rules = """
Citation rules:
- Every factual claim MUST end with an inline citation using the bracketed number(s) of the Current Context chunk(s) it came from, e.g. "The base temperature is 1.6 K [2]."
- Place the citation immediately after the sentence or clause it supports. Cite multiple chunks when a claim draws on several, e.g. "[1][3]".
- Only use numbers that appear in the Current Context. Never invent a citation number, and do not cite the Previous Context.
- Do not add a separate reference list at the end; the interface renders one from your inline markers."""

    if not USE_DOMAIN_PROMPT:
        system_message = """You are a helpful assistant. Answer the user's question using the provided context.
""" + citation_rules + """

Current Context:
{context}

Previous Context (if relevant):
{previous_context}"""
    else:
        system_message = """You are an expert technical assistant for the attoDRY cryostat system and related laboratory instruments (attoDRY2100, ANC300 positioners, FlexPositioners, magnet power supplies, and associated equipment).

Your knowledge comes exclusively from the retrieved context below — a curated set of equipment manuals, specification sheets, and technical support communications.

Rules:
- Be concise. Match your answer length to what the question requires — a single factual question warrants a short, direct answer.
- If the answer requires multiple steps, list them clearly in order.
- When you reference information, name the source document if it is identifiable from the context (e.g., "According to the attoDRY2100 Manual..."). Do not fabricate page numbers or section references that do not appear verbatim in the retrieved text.
- If information comes from multiple sources, synthesize it and note each source document.
- If the retrieved context does not contain enough information to answer fully, say so explicitly rather than guessing.
- For questions about email communications (documents starting with "UHD"), summarize the relevant technical content from the retrieved text.
- Always use the retrieved context as your source of truth. Do not rely on any information not present in the context, even if it seems like common knowledge. If the context does not contain the answer, claim ignorance and say you don't know.
""" + citation_rules + """

Current Context:
{context}

Previous Context (if relevant):
{previous_context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    
    # Create the chain with direct input
    rag_chain = prompt | pro_llm | StrOutputParser()
    
    # Debug: Show what's being sent to the LLM
    input_data = {
        "context": context_str,
        "previous_context": previous_context_str,
        "question": effective_query,
        "history": conversation_history.get_messages()
    }
    
    print(f"DEBUG: Input to LLM chain:")
    print(f"  - Question: {effective_query}")
    print(f"  - Context length: {len(input_data['context'])} chars")
    print(f"  - Previous context length: {len(input_data['previous_context'])} chars")
    print(f"  - History messages: {len(input_data['history'])}")
    print(f"DEBUG: Actual context being sent: {context_str[:300]}...")
    
    if status_callback:
        status_callback("generating", "Generating answer...")
    
    # Get answer
    answer = rag_chain.invoke(input_data)
    
    print(f"DEBUG: LLM Response: {answer[:200]}...")
    
    # Sources are the numbered citation list, so the "References" panel lines up
    # exactly with the inline [n] markers in the answer.
    sources = citations

    # Update conversation history
    conversation_history.add_user_message(query)
    conversation_history.add_ai_message(answer)

    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "images": images,
        "debug_info": debug_info,
        "needs_clarification": False,
        "retrieved_docs": docs
    }