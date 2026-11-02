import json
import pickle

import bm25s
import faiss
import numpy as np
from dotenv import load_dotenv

from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

from beir_scifact_rag.config import (
    DB_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    RERANKER_MODEL_NAME,
    TOP_K,
    CANDIDATE_K,
    RERANK_CANDIDATE_K,
    RRF_K,
)

# ============================================================
# Configuration
# ============================================================

load_dotenv()

RERANKER_BACKEND = "crossencoder"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# Load saved data
# ============================================================

with open(DB_DIR / "chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

with open(DB_DIR / "documents.json", encoding="utf-8") as f:
    documents = json.load(f)

with open(DB_DIR / "bm25.pkl", "rb") as f:
    saved = pickle.load(f)

bm25_index = saved["bm25_index"]

with open(DB_DIR / "config.json", encoding="utf-8") as f:
    index_config = json.load(f)

dense_document_index = faiss.read_index(
    str(DB_DIR / "dense_documents.faiss")
)

print("Loaded chunks:", len(chunks))
print("Loaded documents:", len(documents))
print("Dense document vectors:", dense_document_index.ntotal)
print("Embedding model:", EMBEDDING_MODEL_NAME)
print("Reranker backend:", RERANKER_BACKEND)

if index_config.get("embedding_model") != EMBEDDING_MODEL_NAME:
    raise RuntimeError(
        "The saved dense index was built with "
        f"{index_config.get('embedding_model')!r}, but the configured model is "
        f"{EMBEDDING_MODEL_NAME!r}. Rebuild the index with build_index.py."
    )

if dense_document_index.ntotal != len(documents):
    raise ValueError(
        "Dense index/document count mismatch: "
        f"{dense_document_index.ntotal} vectors versus "
        f"{len(documents)} documents. Rebuild the dense index "
        "or restore the matching documents.json."
    )


# ============================================================
# Optional local CrossEncoder
# ============================================================

cross_encoder = None

if RERANKER_BACKEND == "crossencoder":
    cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)
    print("Loaded local CrossEncoder:", RERANKER_MODEL_NAME)


# ============================================================
# Document lookup
# ============================================================

document_lookup = {
    str(doc["doc_id"]): doc
    for doc in documents
}


def get_document_id(item):
    """Get the original document ID from a chunk or document."""

    value = item.get("doc_id")

    if value is None:
        raise KeyError(
            "Retrieved item has no doc_id. Ensure chunks.json "
            "contains the original document ID for every chunk."
        )

    return str(value)


def get_full_document_text(doc):
    """Return the text used to represent a complete document."""

    text = doc.get("full_text")

    if text:
        return str(text)

    title = str(doc.get("title") or "")
    body = str(doc.get("text") or "")

    return f"{title}\n{body}".strip()


# ============================================================
# Local SentenceTransformer embedding
# ============================================================

def embed_texts(texts, batch_size=16):
    """
    Generate embeddings locally through SentenceTransformers.

    The returned vectors are L2-normalized to match a
    cosine-similarity FAISS IndexFlatIP index.

    The embedding model must match the model used to build
    dense_documents.faiss.
    """

    if not texts:
        return np.empty((0, dense_document_index.d), dtype=np.float32)

    embeddings = embedding_model.encode(
        [str(text) for text in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        truncate_dim=EMBEDDING_DIMENSION,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    if embeddings.shape[1] != dense_document_index.d:
        raise ValueError(
            f"Local embedding dimension {embeddings.shape[1]} does not match FAISS index "
            f"dimension {dense_document_index.d}; rebuild the index."
        )
    return embeddings


# ============================================================
# BM25 chunk retrieval
# ============================================================

def bm25_search(query, k=CANDIDATE_K):
    """Return the top BM25 chunks."""

    if not chunks or k <= 0:
        return []

    query_tokens = bm25s.tokenize(
        [query],
        stopwords="en",
    )

    k = min(k, len(chunks))

    results, scores = bm25_index.retrieve(
        query_tokens,
        k=k,
    )

    output = []

    for idx, score in zip(results[0], scores[0]):
        idx = int(idx)

        if idx < 0 or idx >= len(chunks):
            continue

        item = dict(chunks[idx])
        item["doc_id"] = get_document_id(item)
        item["result_type"] = "chunk"
        item["score"] = float(score)

        output.append(item)

    return output


# ============================================================
# Dense complete-document retrieval
# ============================================================

def dense_search(query, k=CANDIDATE_K):
    """Retrieve complete documents from the dense FAISS index."""

    if dense_document_index.ntotal == 0 or k <= 0:
        return []

    query_embedding = embed_texts([query], batch_size=1)

    k = min(k, dense_document_index.ntotal)

    scores, indices = dense_document_index.search(
        np.ascontiguousarray(query_embedding),
        k,
    )

    output = []

    for idx, score in zip(indices[0], scores[0]):
        idx = int(idx)

        if idx < 0 or idx >= len(documents):
            continue

        doc = documents[idx]

        output.append({
            "doc_id": str(doc["doc_id"]),
            "title": doc.get("title", ""),
            "text": doc.get("text", ""),
            "full_text": get_full_document_text(doc),
            "result_type": "document",
            "score": float(score),
        })

    return output


# ============================================================
# Reciprocal Rank Fusion
# ============================================================

def hybrid_search(
    query,
    k=TOP_K,
    candidate_k=CANDIDATE_K,
    rrf_k=RRF_K,
):
    """
    Fuse BM25 chunk retrieval and dense document retrieval
    at the original-document level.

    RRF(doc) = sum_r 1 / (rrf_k + rank_r(doc))

    Ranks start at 1. Multiple chunks belonging to the same
    document contribute only once to the BM25 ranking, using
    the best rank of that document's chunks.
    """

    if k <= 0 or candidate_k <= 0:
        return []

    bm25_results = bm25_search(query, k=candidate_k)
    dense_results = dense_search(query, k=candidate_k)

    fused_scores = {}
    best_chunk_by_doc = {}
    best_chunk_rank = {}

    # ------------------------------------------
    # BM25: aggregate chunks into document ranks
    # ------------------------------------------

    for rank, result in enumerate(bm25_results, start=1):
        doc_id = get_document_id(result)

        if (
            doc_id not in best_chunk_rank
            or rank < best_chunk_rank[doc_id]
        ):
            best_chunk_rank[doc_id] = rank
            best_chunk_by_doc[doc_id] = result

    for doc_id, rank in best_chunk_rank.items():
        fused_scores[doc_id] = (
            fused_scores.get(doc_id, 0.0)
            + 1.0 / (rrf_k + rank)
        )

    # ------------------------------------------
    # Dense: one rank per complete document
    # ------------------------------------------

    for rank, result in enumerate(dense_results, start=1):
        doc_id = get_document_id(result)

        fused_scores[doc_id] = (
            fused_scores.get(doc_id, 0.0)
            + 1.0 / (rrf_k + rank)
        )

    # ------------------------------------------
    # Sort by RRF score
    # ------------------------------------------

    ranked_ids = sorted(
        fused_scores,
        key=lambda doc_id: fused_scores[doc_id],
        reverse=True,
    )[:k]

    output = []

    for doc_id in ranked_ids:
        doc = document_lookup.get(doc_id)

        if doc is None:
            continue

        item = {
            "doc_id": doc_id,
            "title": doc.get("title", ""),
            "text": doc.get("text", ""),
            "full_text": get_full_document_text(doc),
            "result_type": "document",
            "rrf_score": float(fused_scores[doc_id]),
        }

        # Retain the best-matching BM25 chunk for evidence/debugging.
        if doc_id in best_chunk_by_doc:
            item["bm25_chunk"] = dict(best_chunk_by_doc[doc_id])

        output.append(item)

    return output


# ============================================================
# Local reranking
# ============================================================

# ============================================================
# Local CrossEncoder reranking
# ============================================================

def crossencoder_rerank(query, results, k=TOP_K):
    """Rerank documents using a local SentenceTransformers CrossEncoder."""

    if not results or k <= 0:
        return []

    if cross_encoder is None:
        raise RuntimeError(
            "The local CrossEncoder is not loaded. "
            "Set RERANKER_BACKEND=crossencoder before starting the script."
        )

    pairs = []

    for result in results:
        title = str(result.get("title") or "")
        text = get_full_document_text(result)
        pairs.append((query, f"{title}\n{text}".strip()))

    scores = cross_encoder.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):
        item = dict(result)
        item["rerank_score"] = float(score)
        item["reranker"] = "crossencoder"
        reranked.append(item)

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return reranked[:k]


# ============================================================
# Reranker selection
# ============================================================

def rerank_results(query, results, k=TOP_K, backend=None):
    """Rerank candidates with the configured local CrossEncoder."""

    backend = (backend or RERANKER_BACKEND).lower()

    if backend == "crossencoder":
        return crossencoder_rerank(query, results, k=k)

    raise ValueError("Only the local 'crossencoder' reranker is supported.")


# ============================================================
# Hybrid retrieval + reranking
# ============================================================

def hybrid_rerank_search(
    query,
    k=TOP_K,
    candidate_k=RERANK_CANDIDATE_K,
    reranker_backend=None,
):
    """
    Retrieve document-level candidates with RRF, then rerank them.

    candidate_k should generally be greater than k so the reranker
    can reorder a larger candidate set.
    """

    candidate_k = min(50, max(20, int(candidate_k)))

    candidates = hybrid_search(
        query,
        k=candidate_k,
        candidate_k=candidate_k,
    )

    return rerank_results(
        query,
        candidates,
        k=k,
        backend=reranker_backend,
    )


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    query = "How does the system perform document retrieval?"

    tests = [
        ("BM25 Chunks", lambda q: bm25_search(q, k=5)),
        ("Dense Complete Documents", lambda q: dense_search(q, k=5)),
        ("Hybrid RRF", lambda q: hybrid_search(q, k=5)),
        (
            "Hybrid + Local BGE Reranker",
            lambda q: hybrid_rerank_search(
                q,
                k=5,
                candidate_k=CANDIDATE_K,
                reranker_backend="crossencoder",
            ),
        ),
    ]

    # Run the local CrossEncoder baseline separately when desired.
    if RERANKER_BACKEND == "crossencoder":
        tests[-1] = (
            "Hybrid + Local CrossEncoder",
            lambda q: hybrid_rerank_search(
                q,
                k=5,
                candidate_k=CANDIDATE_K,
                reranker_backend="crossencoder",
            ),
        )

    for name, search_fn in tests:
        print(f"\n{'=' * 70}")
        print(name)

        try:
            results = search_fn(query)
        except Exception as exc:
            print(f"Search failed: {exc}")
            continue

        for rank, result in enumerate(results, start=1):
            score = result.get(
                "rerank_score",
                result.get(
                    "rrf_score",
                    result.get("score", 0.0),
                ),
            )

            print(
                f"\nRank: {rank} | "
                f"Doc ID: {result['doc_id']} | "
                f"Score: {score:.4f}"
            )
            print("Title:", result.get("title", ""))
            print("Text:", result.get("text", "")[:250])
