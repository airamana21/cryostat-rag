"""
Synthetic Q&A dataset generator for Cryostat RAG evaluation.

Generates question-answer pairs from the cryostat documents by prompting
the LLM to create questions of different types (factual, procedural,
multi-hop, edge-case) and validating them against the source chunks.
"""
import json
import os
import random
import sys
from typing import List, Dict, Tuple, Optional

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import evaluation.env_setup  # noqa: F401 — must precede rag imports

from rag import (
    load_and_split_pdfs,
    VertexAIGeminiLLM,
    download_pdfs_from_gcs,
    GCS_BUCKET_NAME,
    GCS_PDF_PREFIX,
    PDF_CACHE_DIR,
)
from evaluation.eval_config import (
    NUM_FACTUAL_QUESTIONS,
    NUM_PROCEDURAL_QUESTIONS,
    NUM_MULTIHOP_QUESTIONS,
    NUM_EDGE_CASE_QUESTIONS,
    MIN_CHUNK_LENGTH,
    VALIDATION_RETRIES,
    SYNTHETIC_QA_PATH,
)

# ─── Prompts for question generation ────────────────────────────────────────

FACTUAL_PROMPT = """You are generating evaluation questions for a RAG system about cryogenic equipment.

Given the following text from a technical document, generate exactly ONE specific factual question whose answer can be found directly in the text. The question should ask about a specific fact, number, name, specification, or detail.

Text:
{chunk_text}

Source: {source} (Page {page})

Respond in this exact JSON format (no markdown, no code fences):
{{"question": "your specific factual question here", "answer": "the exact answer from the text", "reasoning": "why this is a good factual question"}}"""

PROCEDURAL_PROMPT = """You are generating evaluation questions for a RAG system about cryogenic equipment.

Given the following text from a technical document that describes a procedure or process, generate exactly ONE "how-to" question that requires understanding the steps or procedure described.

Text:
{chunk_text}

Source: {source} (Page {page})

Respond in this exact JSON format (no markdown, no code fences):
{{"question": "your how-to/procedural question here", "answer": "a complete answer describing the procedure from the text", "reasoning": "why this requires procedural understanding"}}"""

MULTIHOP_PROMPT = """You are generating evaluation questions for a RAG system about cryogenic equipment.

Given the following two text passages from technical documents, generate exactly ONE question that requires information from BOTH passages to answer correctly. The question should require synthesizing or connecting information across both texts.

Passage 1:
{chunk_text_1}
Source: {source_1} (Page {page_1})

Passage 2:
{chunk_text_2}
Source: {source_2} (Page {page_2})

Respond in this exact JSON format (no markdown, no code fences):
{{"question": "your multi-hop question here", "answer": "answer that draws on both passages", "reasoning": "what information from each passage is needed"}}"""

EDGE_CASE_TEMPLATES = [
    {"question": "what", "category": "vague_single_word", "expected_behavior": "should_clarify"},
    {"question": "help", "category": "vague_single_word", "expected_behavior": "should_clarify"},
    {"question": "tell me about the cryostat", "category": "vague_general", "expected_behavior": "should_clarify"},
    {"question": "explain", "category": "vague_single_word", "expected_behavior": "should_clarify"},
    {"question": "info", "category": "vague_single_word", "expected_behavior": "should_clarify"},
    {"question": "specs", "category": "vague_single_word", "expected_behavior": "should_clarify"},
    {"question": "?", "category": "vague_single_word", "expected_behavior": "should_clarify"},
    {"question": "What is the temperature?", "category": "ambiguous_but_specific_word", "expected_behavior": "should_answer"},
    {"question": "Tell me more about that", "category": "follow_up_no_context", "expected_behavior": "follow_up"},
    {"question": "What else?", "category": "follow_up_no_context", "expected_behavior": "follow_up"},
    {"question": "Can you elaborate?", "category": "follow_up_no_context", "expected_behavior": "follow_up"},
    {"question": "continue", "category": "follow_up_no_context", "expected_behavior": "follow_up"},
    {"question": "What is the serial number of the flux capacitor?", "category": "unanswerable", "expected_behavior": "should_not_hallucinate"},
    {"question": "How do I order replacement parts from Amazon?", "category": "out_of_scope", "expected_behavior": "should_not_hallucinate"},
    {"question": "What color is the cryostat's unicorn logo?", "category": "unanswerable", "expected_behavior": "should_not_hallucinate"},
    {"question": "How do I install the model 5000 compressor?", "category": "procedural_with_model", "expected_behavior": "should_answer"},
    {"question": "What are the voltage specs?", "category": "specific_short", "expected_behavior": "should_answer"},
    {"question": "details about the serial number", "category": "specific_keyword", "expected_behavior": "should_answer"},
    {"question": "How do I set up the system?", "category": "procedural_general", "expected_behavior": "should_answer"},
    {"question": "What emails discuss temperature issues?", "category": "doc_type_filter", "expected_behavior": "should_filter_emails"},
]


def _parse_llm_json(response: str) -> Optional[Dict]:
    """Parse JSON from LLM response, handling common formatting issues."""
    text = response.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None


def _select_chunks_for_category(
    docs: List, category: str, num_needed: int, min_length: int = MIN_CHUNK_LENGTH
) -> List:
    """Select appropriate chunks for a given question category."""
    eligible = [d for d in docs if len(d.page_content.strip()) >= min_length]
    if not eligible:
        return []

    if category == "factual":
        # Prefer chunks with numbers, model names, specifications
        scored = []
        for d in eligible:
            content = d.page_content
            score = sum(c.isdigit() for c in content) + content.count("model") + content.count("serial")
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = [d for _, d in scored[:max(len(scored) // 2, num_needed * 3)]]
    elif category == "procedural":
        # Prefer chunks with procedural language
        proc_words = {"step", "install", "connect", "remove", "attach", "tighten", "adjust", "procedure"}
        scored = []
        for d in eligible:
            words = set(d.page_content.lower().split())
            score = len(words & proc_words)
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = [d for _, d in scored[:max(len(scored) // 2, num_needed * 3)]]
    else:
        candidates = eligible

    random.shuffle(candidates)
    return candidates[:num_needed * 2]  # Return extra for failures


def generate_factual_questions(
    fine_docs: List, llm: VertexAIGeminiLLM, num: int = NUM_FACTUAL_QUESTIONS
) -> List[Dict]:
    """Generate factual Q&A pairs from fine-grained chunks."""
    questions = []
    chunks = _select_chunks_for_category(fine_docs, "factual", num)

    for chunk in tqdm(chunks, desc="Generating factual questions"):
        if len(questions) >= num:
            break

        prompt = FACTUAL_PROMPT.format(
            chunk_text=chunk.page_content,
            source=chunk.metadata.get("source", "unknown"),
            page=chunk.metadata.get("page", 0),
        )

        for attempt in range(VALIDATION_RETRIES + 1):
            try:
                response = llm._call(prompt)
                parsed = _parse_llm_json(response)
                if parsed and "question" in parsed and "answer" in parsed:
                    questions.append({
                        "id": f"factual_{len(questions):03d}",
                        "question": parsed["question"],
                        "reference_answer": parsed["answer"],
                        "category": "factual",
                        "source_chunks": [{
                            "doc_id": chunk.metadata.get("doc_id", chunk.metadata.get("source", "")),
                            "chunk_index": chunk.metadata.get("chunk_index", -1),
                            "page": chunk.metadata.get("page", 0),
                            "source": chunk.metadata.get("source", ""),
                        }],
                        "expected_db": "fine",
                        "expected_doc_type": chunk.metadata.get("doc_type", "manual"),
                    })
                    break
            except Exception as e:
                if attempt == VALIDATION_RETRIES:
                    print(f"  Failed to generate factual Q from chunk: {e}")

    return questions


def generate_procedural_questions(
    coarse_docs: List, llm: VertexAIGeminiLLM, num: int = NUM_PROCEDURAL_QUESTIONS
) -> List[Dict]:
    """Generate procedural Q&A pairs from coarse-grained chunks."""
    questions = []
    chunks = _select_chunks_for_category(coarse_docs, "procedural", num)

    for chunk in tqdm(chunks, desc="Generating procedural questions"):
        if len(questions) >= num:
            break

        prompt = PROCEDURAL_PROMPT.format(
            chunk_text=chunk.page_content,
            source=chunk.metadata.get("source", "unknown"),
            page=chunk.metadata.get("page", 0),
        )

        for attempt in range(VALIDATION_RETRIES + 1):
            try:
                response = llm._call(prompt)
                parsed = _parse_llm_json(response)
                if parsed and "question" in parsed and "answer" in parsed:
                    questions.append({
                        "id": f"procedural_{len(questions):03d}",
                        "question": parsed["question"],
                        "reference_answer": parsed["answer"],
                        "category": "procedural",
                        "source_chunks": [{
                            "doc_id": chunk.metadata.get("doc_id", chunk.metadata.get("source", "")),
                            "chunk_index": chunk.metadata.get("chunk_index", -1),
                            "page": chunk.metadata.get("page", 0),
                            "source": chunk.metadata.get("source", ""),
                        }],
                        "expected_db": "coarse",
                        "expected_doc_type": chunk.metadata.get("doc_type", "manual"),
                    })
                    break
            except Exception as e:
                if attempt == VALIDATION_RETRIES:
                    print(f"  Failed to generate procedural Q from chunk: {e}")

    return questions


def generate_multihop_questions(
    fine_docs: List, llm: VertexAIGeminiLLM, num: int = NUM_MULTIHOP_QUESTIONS
) -> List[Dict]:
    """Generate multi-hop Q&A pairs requiring information from multiple chunks."""
    questions = []
    eligible = [d for d in fine_docs if len(d.page_content.strip()) >= MIN_CHUNK_LENGTH]

    # Group chunks by document
    doc_groups = {}
    for d in eligible:
        doc_id = d.metadata.get("doc_id", d.metadata.get("source", "unknown"))
        doc_groups.setdefault(doc_id, []).append(d)

    # Generate pairs from same or different documents
    pairs = []
    doc_ids = list(doc_groups.keys())
    for doc_id in doc_ids:
        chunks = doc_groups[doc_id]
        if len(chunks) >= 2:
            for _ in range(min(num, len(chunks) // 2)):
                pair = random.sample(chunks, 2)
                pairs.append(pair)

    random.shuffle(pairs)

    for chunk1, chunk2 in tqdm(pairs[:num * 2], desc="Generating multi-hop questions"):
        if len(questions) >= num:
            break

        prompt = MULTIHOP_PROMPT.format(
            chunk_text_1=chunk1.page_content,
            source_1=chunk1.metadata.get("source", "unknown"),
            page_1=chunk1.metadata.get("page", 0),
            chunk_text_2=chunk2.page_content,
            source_2=chunk2.metadata.get("source", "unknown"),
            page_2=chunk2.metadata.get("page", 0),
        )

        for attempt in range(VALIDATION_RETRIES + 1):
            try:
                response = llm._call(prompt)
                parsed = _parse_llm_json(response)
                if parsed and "question" in parsed and "answer" in parsed:
                    questions.append({
                        "id": f"multihop_{len(questions):03d}",
                        "question": parsed["question"],
                        "reference_answer": parsed["answer"],
                        "category": "multi_hop",
                        "source_chunks": [
                            {
                                "doc_id": chunk1.metadata.get("doc_id", chunk1.metadata.get("source", "")),
                                "chunk_index": chunk1.metadata.get("chunk_index", -1),
                                "page": chunk1.metadata.get("page", 0),
                                "source": chunk1.metadata.get("source", ""),
                            },
                            {
                                "doc_id": chunk2.metadata.get("doc_id", chunk2.metadata.get("source", "")),
                                "chunk_index": chunk2.metadata.get("chunk_index", -1),
                                "page": chunk2.metadata.get("page", 0),
                                "source": chunk2.metadata.get("source", ""),
                            },
                        ],
                        "expected_db": "fine",
                        "expected_doc_type": None,
                    })
                    break
            except Exception as e:
                if attempt == VALIDATION_RETRIES:
                    print(f"  Failed to generate multi-hop Q: {e}")

    return questions


def generate_edge_case_questions(num: int = NUM_EDGE_CASE_QUESTIONS) -> List[Dict]:
    """Generate edge-case questions from predefined templates."""
    templates = EDGE_CASE_TEMPLATES[:]
    random.shuffle(templates)
    questions = []

    for i, template in enumerate(templates[:num]):
        questions.append({
            "id": f"edge_{i:03d}",
            "question": template["question"],
            "reference_answer": None,  # No reference for edge cases
            "category": "edge_case",
            "subcategory": template["category"],
            "expected_behavior": template["expected_behavior"],
            "source_chunks": [],
            "expected_db": None,
            "expected_doc_type": None,
        })

    return questions


def generate_synthetic_dataset(
    pdf_folder: str = None,
    output_path: str = SYNTHETIC_QA_PATH,
    num_factual: int = NUM_FACTUAL_QUESTIONS,
    num_procedural: int = NUM_PROCEDURAL_QUESTIONS,
    num_multihop: int = NUM_MULTIHOP_QUESTIONS,
    num_edge: int = NUM_EDGE_CASE_QUESTIONS,
) -> List[Dict]:
    """Generate the complete synthetic evaluation dataset.

    Args:
        pdf_folder: Path to folder containing PDFs. If None, downloads from GCS.
        output_path: Where to save the generated dataset JSON.
        num_factual: Number of factual questions to generate.
        num_procedural: Number of procedural questions to generate.
        num_multihop: Number of multi-hop questions to generate.
        num_edge: Number of edge-case questions to include.

    Returns:
        List of all generated Q&A dicts.
    """
    # Load documents
    if pdf_folder is None:
        print("Downloading PDFs from GCS...")
        pdf_folder = download_pdfs_from_gcs(GCS_BUCKET_NAME, GCS_PDF_PREFIX, PDF_CACHE_DIR)

    print("Loading and splitting PDFs...")
    fine_docs, coarse_docs = load_and_split_pdfs(pdf_folder)
    print(f"  Fine chunks: {len(fine_docs)}, Coarse chunks: {len(coarse_docs)}")

    llm = VertexAIGeminiLLM(model_name="gemini-2.5-pro")
    all_questions = []

    # Generate each category
    print(f"\n--- Generating {num_factual} factual questions ---")
    all_questions.extend(generate_factual_questions(fine_docs, llm, num_factual))

    print(f"\n--- Generating {num_procedural} procedural questions ---")
    all_questions.extend(generate_procedural_questions(coarse_docs, llm, num_procedural))

    print(f"\n--- Generating {num_multihop} multi-hop questions ---")
    all_questions.extend(generate_multihop_questions(fine_docs, llm, num_multihop))

    print(f"\n--- Adding {num_edge} edge-case questions ---")
    all_questions.extend(generate_edge_case_questions(num_edge))

    # Summary
    print(f"\nDataset generation complete:")
    categories = {}
    for q in all_questions:
        cat = q["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print(f"  Total: {len(all_questions)}")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")

    return all_questions


def load_synthetic_dataset(path: str = SYNTHETIC_QA_PATH) -> List[Dict]:
    """Load a previously generated synthetic dataset."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    generate_synthetic_dataset()
