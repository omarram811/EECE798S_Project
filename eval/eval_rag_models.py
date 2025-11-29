# eval/eval_rag_models.py

"""
eval_rag_models.py

Evaluation harness for your Agentic Systems TA project.

- Loads golden_set.json from eval/golden_set.json
- Uses a local document RAG index built from eval/documents/
- Evaluates two providers by default:
    * OpenAI  (chat model: gpt-5-nano, embedding: text-embedding-3-small)
    * Gemini  (chat model: gemini-2.5-flash, embedding: models/text-embedding-004)

For each (question, model) pair, we record:
  - retrieval quality (does top-1/top-k document match the expected slide deck?)
  - answer text
  - lexical F1 vs golden answer
  - simple semantic similarity via embeddings
  - latency and token usage (if the API returns it)
  - approximate cost (if you set *_COST_PER_1K env vars)

Results are written as JSON arrays to:
  eval/results/openai_gpt-5-nano_results.json
  eval/results/gemini_gemini-2.5-flash_results.json

You can later aggregate them with summarize_eval_results.py
"""

from __future__ import annotations

import json
import math
import os
import string
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pypdf import PdfReader

# These imports will use the same client libraries as the main app
from openai import OpenAI
import google.generativeai as genai

import dotenv
dotenv.load_dotenv()

# ----------------------------------------------------------------------
# Paths & basic config
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
DOCS_DIR = EVAL_DIR / "docs"
GOLDEN_PATH = EVAL_DIR / "golden_set.json"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = int(os.getenv("RAG_TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))
REPEATS_PER_QUESTION = int(os.getenv("EVAL_REPEATS", "1"))
TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE", "0.0"))

# Pricing config (tokens per 1K); set these in your .env if you want cost estimates
OPENAI_INPUT_COST_PER_1K = float(os.getenv("OPENAI_INPUT_COST_PER_1K", "0"))
OPENAI_OUTPUT_COST_PER_1K = float(os.getenv("OPENAI_OUTPUT_COST_PER_1K", "0"))
GEMINI_INPUT_COST_PER_1K = float(os.getenv("GEMINI_INPUT_COST_PER_1K", "0"))
GEMINI_OUTPUT_COST_PER_1K = float(os.getenv("GEMINI_OUTPUT_COST_PER_1K", "0"))

# Model names (match what you gave me)
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------


def _normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not s:
        return ""
    s = s.lower()
    table = str.maketrans("", "", string.punctuation)
    s = s.translate(table)
    parts = s.split()
    return " ".join(parts)


def f1_score(pred: str, gold: str) -> float:
    """Simple token-level F1 between predicted and golden answers."""
    pred_norm = _normalize_text(pred)
    gold_norm = _normalize_text(gold)
    if not pred_norm and not gold_norm:
        return 1.0
    if not pred_norm or not gold_norm:
        return 0.0

    p_tokens = pred_norm.split()
    g_tokens = gold_norm.split()

    p_set = set(p_tokens)
    g_set = set(g_tokens)
    common = p_set.intersection(g_set)
    if not common:
        return 0.0
    prec = len(common) / max(len(p_set), 1)
    rec = len(common) / max(len(g_set), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Shapes must match for cosine similarity")
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_golden_set(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Document loading & indexing
# ----------------------------------------------------------------------


@dataclass
class DocChunk:
    document: str      # filename only
    chunk_id: int
    text: str
    embedding: np.ndarray


class DocIndex:
    """
    Very lightweight embedding-based index for eval only.
    We do brute-force cosine search over all chunks.
    """

    def __init__(self, docs_dir: Path, embed_fn):
        """
        embed_fn: callable(List[str]) -> List[List[float]]
        """
        self.docs_dir = docs_dir
        self.embed_fn = embed_fn
        self.chunks: List[DocChunk] = []
        self._embedding_matrix: np.ndarray | None = None
        self._build_index()

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        reader = PdfReader(str(pdf_path))
        parts = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            parts.append(txt)
        return "\n".join(parts)

    def _chunk_text(self, text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
        """
        Simple character-based sliding window chunking.
        Enough for this eval harness.
        """
        text = text.strip()
        if not text:
            return []
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + chunk_size)
            chunk = text[start:end]
            chunks.append(chunk)
            if end == n:
                break
            start = max(end - overlap, 0)
        return chunks

    def _build_index(self):
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")

        pdf_files = sorted(self.docs_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.docs_dir}")

        all_chunk_texts: List[str] = []
        meta: List[Tuple[str, int]] = []

        print(f"[DocIndex] Building index from {len(pdf_files)} PDF files in {self.docs_dir} ...")

        for pdf_path in pdf_files:
            fname = pdf_path.name
            text = self._extract_pdf_text(pdf_path)
            chunks = self._chunk_text(text)
            print(f"[DocIndex] {fname}: {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                all_chunk_texts.append(chunk)
                meta.append((fname, i))

        print(f"[DocIndex] Embedding {len(all_chunk_texts)} chunks ...")
        embeddings = self.embed_fn(all_chunk_texts)
        if not embeddings:
            raise RuntimeError("Embedding function returned no embeddings")

        emb_arr = np.array(embeddings, dtype="float32")

        # Some providers (e.g., Gemini embed_content with batched input) return
        # an extra leading batch dimension, e.g. shape (1, N, D). Flatten that.
        if emb_arr.ndim == 3 and emb_arr.shape[0] == 1:
            emb_arr = emb_arr[0]

        if emb_arr.ndim != 2:
            raise RuntimeError(f"Expected 2D embedding array, got shape {emb_arr.shape}")


        self._embedding_matrix = emb_arr

        self.chunks = []
        for (fname, cid), emb, text in zip(meta, emb_arr, all_chunk_texts):
            self.chunks.append(DocChunk(document=fname, chunk_id=cid, text=text, embedding=emb))

        print(f"[DocIndex] Index ready with {len(self.chunks)} chunks.")

    def retrieve(self, query: str, k: int = 5) -> List[DocChunk]:
        if self._embedding_matrix is None:
            raise RuntimeError("Index not built")

        q_emb = np.array(self.embed_fn([query])[0], dtype="float32")
        sims = self._embedding_matrix @ q_emb / (
            np.linalg.norm(self._embedding_matrix, axis=1) * (np.linalg.norm(q_emb) + 1e-8)
        )
        # Handle potential NaNs
        sims = np.nan_to_num(sims, nan=0.0)
        idxs = np.argsort(-sims)[:k]
        out = []
        for idx in idxs:
            ch = self.chunks[int(idx)]
            out.append(ch)
        return out


# ----------------------------------------------------------------------
# Providers used for evaluation (non-streaming)
# ----------------------------------------------------------------------


class OpenAIQA:
    def __init__(self, model: str, embed_model: str):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY env var is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embed_model = embed_model

    # For document index + semantic similarity
    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in resp.data]

    def answer(self, question: str, context: str) -> Tuple[str, Dict[str, Any]]:
        system_prompt = (
            "You are a helpful teaching assistant for the EECE 798S Agentic Systems course.\n"
            "Use ONLY the provided context from the official course materials to answer.\n"
            "If the context is not sufficient, say explicitly: 'I don't know from the course materials'."
        )
        user_prompt = (
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Answer concisely in 1–3 sentences."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE,
        )
        dt = time.perf_counter() - t0

        ans = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)

        # Try to support both old and new token-usage schemas
        input_tokens = None
        output_tokens = None
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)

        # Cost (if you configured prices)
        total_input_cost = 0.0
        total_output_cost = 0.0
        if input_tokens is not None and OPENAI_INPUT_COST_PER_1K > 0:
            total_input_cost = (input_tokens / 1000.0) * OPENAI_INPUT_COST_PER_1K
        if output_tokens is not None and OPENAI_OUTPUT_COST_PER_1K > 0:
            total_output_cost = (output_tokens / 1000.0) * OPENAI_OUTPUT_COST_PER_1K

        meta = {
            "latency_seconds": dt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0),
            "input_cost": total_input_cost,
            "output_cost": total_output_cost,
            "total_cost": total_input_cost + total_output_cost,
        }
        return ans, meta


class GeminiQA:
    def __init__(self, model: str, embed_model: str):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY env var is not set")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.embed_model = embed_model
        self.gen_model = genai.GenerativeModel(model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Robust embedding wrapper for Gemini.

        We explicitly call embed_content *per text* so we always get
        a single embedding vector per input, instead of a big (1, N, D)
        batch that messes with shapes.
        """
        embeddings: List[List[float]] = []

        for t in texts:
            out = genai.embed_content(model=self.embed_model, content=t)

            # Newer clients usually return: {"embedding": [floats...]}
            if isinstance(out, dict):
                emb = out.get("embedding")
                if emb is None:
                    raise RuntimeError("Gemini embed_content response has no 'embedding' key")

                # Sometimes "embedding" itself can be a dict with "values" / "value"
                if isinstance(emb, dict):
                    emb = emb.get("values") or emb.get("value")
                    if emb is None:
                        raise RuntimeError("Gemini embed_content embedding dict missing 'values'/'value'")

                if not isinstance(emb, list):
                    raise RuntimeError(f"Unexpected embedding type from Gemini: {type(emb)}")

                embeddings.append(emb)
                continue

            # Fallback: some clients might return an object with an .embedding attr
            emb_attr = getattr(out, "embedding", None)
            if emb_attr is not None:
                emb = emb_attr
                if isinstance(emb, dict):
                    emb = emb.get("values") or emb.get("value")
                embeddings.append(emb)
                continue

            raise RuntimeError(f"Unexpected Gemini embedding response type: {type(out)}")

        return embeddings

    def answer(self, question: str, context: str) -> Tuple[str, Dict[str, Any]]:
        prompt = (
            "You are a helpful teaching assistant for the EECE 798S Agentic Systems course.\n"
            "Use ONLY the provided context from the official course materials to answer.\n"
            "If the context is not sufficient, say explicitly: 'I don't know from the course materials'.\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Answer concisely in 1–3 sentences."
        )

        t0 = time.perf_counter()
        resp = self.gen_model.generate_content(prompt)
        dt = time.perf_counter() - t0

        ans = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)

        # Gemini usage metadata sometimes includes token counts
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = getattr(usage, "candidates_token_count", None) if usage else None

        total_input_cost = 0.0
        total_output_cost = 0.0
        if input_tokens is not None and GEMINI_INPUT_COST_PER_1K > 0:
            total_input_cost = (input_tokens / 1000.0) * GEMINI_INPUT_COST_PER_1K
        if output_tokens is not None and GEMINI_OUTPUT_COST_PER_1K > 0:
            total_output_cost = (output_tokens / 1000.0) * GEMINI_OUTPUT_COST_PER_1K

        meta = {
            "latency_seconds": dt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0),
            "input_cost": total_input_cost,
            "output_cost": total_output_cost,
            "total_cost": total_input_cost + total_output_cost,
        }
        return ans, meta


# ----------------------------------------------------------------------
# Single-question evaluation
# ----------------------------------------------------------------------


def build_context_from_chunks(chunks: List[DocChunk], max_chars: int) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Join retrieved chunks up to max_chars, and return context + metadata
    useful for debugging / observability.
    """
    parts = []
    used_meta = []
    total_chars = 0

    for ch in chunks:
        text = ch.text.strip().replace("\n", " ")
        if not text:
            continue
        if total_chars + len(text) > max_chars:
            break
        parts.append(text)
        total_chars += len(text)
        used_meta.append(
            {
                "document": ch.document,
                "chunk_id": ch.chunk_id,
            }
        )
    return "\n".join(parts), used_meta


def evaluate_single_question(
    provider_name: str,
    qa,
    index: DocIndex,
    item: Dict[str, Any],
    repeat_id: int,
) -> Dict[str, Any]:
    """
    provider_name: "openai" or "gemini"
    qa: OpenAIQA or GeminiQA instance
    index: DocIndex built with qa.embed
    item: single record from golden_set.json
    """

    question = item["question"]
    gold_answer = item["answer"]
    gold_doc = item["document"]

    retrieved_chunks = index.retrieve(question, k=TOP_K)
    context, used_meta = build_context_from_chunks(retrieved_chunks, MAX_CONTEXT_CHARS)

    # retrieval metrics
    retrieved_docs = [ch.document for ch in retrieved_chunks]
    top1_doc = retrieved_docs[0] if retrieved_docs else None
    retrieval_top1_matches = top1_doc == gold_doc
    any_match_in_topk = gold_doc in retrieved_docs

    # call model
    answer_text, meta = qa.answer(question=question, context=context)

    # lexical metrics
    f1 = f1_score(answer_text, gold_answer)
    exact_match_norm = _normalize_text(answer_text) == _normalize_text(gold_answer)
    contains_match = (
        _normalize_text(gold_answer) in _normalize_text(answer_text)
        or _normalize_text(answer_text) in _normalize_text(gold_answer)
    )
    is_correct = exact_match_norm or contains_match or f1 >= 0.75

    # semantic similarity via embeddings (same provider)
    try:
        embs = qa.embed([gold_answer, answer_text])
        emb_arr = np.array(embs, dtype="float32")

        # Handle providers that wrap the batch in an extra dimension: (1, 2, D)
        if emb_arr.ndim == 3 and emb_arr.shape[0] == 1:
            emb_arr = emb_arr[0]

        if emb_arr.shape[0] < 2:
            raise RuntimeError(f"Expected at least 2 embeddings, got shape {emb_arr.shape}")

        a = emb_arr[0]
        b = emb_arr[1]
        semantic_sim = cosine_similarity(a, b)
    except Exception as e:
        print(f"[WARN] Failed to compute semantic similarity for {provider_name}: {e}")
        semantic_sim = None
    

    result = {
        "provider": provider_name,
        "model": OPENAI_CHAT_MODEL if provider_name == "openai" else GEMINI_CHAT_MODEL,
        "repeat_id": repeat_id,
        "document": gold_doc,
        "question": question,
        "gold_answer": gold_answer,
        "answer": answer_text,
        "retrieval": {
            "top_k": TOP_K,
            "retrieved_docs": retrieved_docs,
            "top1_doc": top1_doc,
            "top1_matches_gold": retrieval_top1_matches,
            "any_match_in_topk": any_match_in_topk,
            "chunks_used": used_meta,
        },
        "metrics": {
            "f1": f1,
            "exact_match_normalized": exact_match_norm,
            "contains_match": contains_match,
            "is_correct": is_correct,
            "semantic_similarity": semantic_sim,
            # latency & cost
            "latency_seconds": meta.get("latency_seconds"),
            "input_tokens": meta.get("input_tokens"),
            "output_tokens": meta.get("output_tokens"),
            "total_tokens": meta.get("total_tokens"),
            "input_cost": meta.get("input_cost"),
            "output_cost": meta.get("output_cost"),
            "total_cost": meta.get("total_cost"),
        },
    }
    return result


# ----------------------------------------------------------------------
# Full run for a given provider
# ----------------------------------------------------------------------


def run_eval_for_provider(provider_name: str, golden: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if provider_name == "openai":
        qa = OpenAIQA(OPENAI_CHAT_MODEL, OPENAI_EMBED_MODEL)
    elif provider_name == "gemini":
        qa = GeminiQA(GEMINI_CHAT_MODEL, GEMINI_EMBED_MODEL)
    else:
        raise ValueError(f"Unknown provider {provider_name}")

    # Build doc index using the provider's embeddings
    index = DocIndex(DOCS_DIR, embed_fn=qa.embed)

    all_results: List[Dict[str, Any]] = []

    for repeat_id in range(REPEATS_PER_QUESTION):
        print(f"[Eval] Provider={provider_name}, repeat {repeat_id+1}/{REPEATS_PER_QUESTION}")
        for i, item in enumerate(golden):
            print(f"[Eval] Q {i+1}/{len(golden)}: {item['question'][:80]}...")
            res = evaluate_single_question(provider_name, qa, index, item, repeat_id=repeat_id)
            all_results.append(res)

    return all_results


def main():
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"Golden set not found at {GOLDEN_PATH}")

    golden = load_golden_set(GOLDEN_PATH)
    print(f"[Eval] Loaded {len(golden)} golden QA pairs from {GOLDEN_PATH}")

    # You can comment out one of these if you only want a single provider
    # providers_to_run = ["openai", "gemini"]
    providers_to_run = ["gemini"]

    for provider_name in providers_to_run:
        print(f"\n========== Evaluating provider: {provider_name} ==========\n")
        results = run_eval_for_provider(provider_name, golden)

        out_name = (
            f"{provider_name}_{OPENAI_CHAT_MODEL}_results.json"
            if provider_name == "openai"
            else f"{provider_name}_{GEMINI_CHAT_MODEL}_results.json"
        )
        out_path = RESULTS_DIR / out_name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[Eval] Wrote {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()
