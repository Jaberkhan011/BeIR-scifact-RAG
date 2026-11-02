import hashlib
import json
import os
import pickle
from pathlib import Path

import bm25s
import faiss
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from beir_scifact_rag.config import (
    CORPUS_PATH,
    DATASET_NAME,
    DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
)


# ============================================================
# Environment and configuration
# ============================================================

load_dotenv()

BATCH_SIZE = max(
    1,
    EMBEDDING_BATCH_SIZE,
)
embedding_model = None


def local_embed_texts(texts, batch_size):
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return embedding_model.encode(
        [str(text) for text in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        truncate_dim=EMBEDDING_DIMENSION,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

# Persistent cache of document embeddings.
EMBEDDING_CACHE_PATH = DB_DIR / "embedding_cache.npz"
EMBEDDING_CACHE_META_PATH = DB_DIR / "embedding_cache.json"

# ============================================================
# Load and clean corpus
# ============================================================

def load_corpus():
    df = pd.read_parquet(CORPUS_PATH)

    required = {"_id", "title", "text"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Corpus must contain columns: {required}"
        )

    df["_id"] = df["_id"].astype(str)

    for col in ["title", "text"]:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df[col] = df[col].replace({
            "NaN": "",
            "nan": "",
            "None": "",
        })

    df = (
        df.drop_duplicates(subset="_id")
        .reset_index(drop=True)
    )

    df = df[
        (df["title"] != "")
        | (df["text"] != "")
    ].reset_index(drop=True)

    return df


# ============================================================
# Prepare complete documents
# ============================================================

def create_documents(df):
    documents = []

    for _, row in df.iterrows():
        doc_id = str(row["_id"])
        title = row["title"]
        text = row["text"]

        full_text = (
            f"{title}\n{text}".strip()
            if title
            else text
        )

        documents.append({
            "doc_id": doc_id,
            "title": title,
            "text": text,
            "full_text": full_text,
        })

    return documents


# ============================================================
# Create chunks for BM25
# ============================================================

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for doc in documents:
        document_chunks = splitter.split_text(
            doc["full_text"]
        )

        for chunk_text in document_chunks:
            chunks.append({
                "chunk_id": len(chunks),
                "doc_id": str(doc["doc_id"]),
                "title": doc["title"],
                "text": chunk_text,
            })

    return chunks


# ============================================================
# Build BM25 chunk index
# ============================================================

def build_bm25(chunks):
    if not chunks:
        raise ValueError(
            "Cannot build BM25: no chunks available."
        )

    tokenized_corpus = bm25s.tokenize(
        [chunk["text"] for chunk in chunks],
        stopwords="en",
    )

    bm25_index = bm25s.BM25()
    bm25_index.index(tokenized_corpus)

    return bm25_index, tokenized_corpus


# ============================================================
# Persistent embedding cache
# ============================================================

def get_cache_fingerprint(documents):
    """
    Generate a fingerprint for the exact embedding inputs.

    The cache is invalidated if the model, document order,
    IDs, or text changes.
    """

    hasher = hashlib.sha256()

    hasher.update(EMBEDDING_MODEL_NAME.encode("utf-8"))

    for doc in documents:
        hasher.update(str(doc["doc_id"]).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(
            str(doc["full_text"]).encode("utf-8")
        )
        hasher.update(b"\0")

    return hasher.hexdigest()


def load_embedding_cache(documents):
    """
    Load completed embeddings if they match the current corpus.

    Returns:
        embeddings_by_index: dict[int, np.ndarray]
        dimension: int | None
    """

    if not (
        EMBEDDING_CACHE_PATH.exists()
        and EMBEDDING_CACHE_META_PATH.exists()
    ):
        return {}, None

    try:
        with open(
            EMBEDDING_CACHE_META_PATH,
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

        expected_fingerprint = get_cache_fingerprint(
            documents
        )

        if metadata.get("fingerprint") != expected_fingerprint:
            print(
                "Embedding cache does not match the current corpus/model. "
                "Starting a fresh cache."
            )
            return {}, None

        with np.load(
            EMBEDDING_CACHE_PATH,
            allow_pickle=False,
        ) as data:
            indices = data["indices"].astype(np.int64)
            vectors = data["vectors"].astype(np.float32)

        if len(indices) != len(vectors):
            print("Invalid embedding cache. Starting fresh.")
            return {}, None

        if vectors.ndim != 2:
            print("Invalid embedding cache shape. Starting fresh.")
            return {}, None

        if not np.isfinite(vectors).all():
            print("Cache contains non-finite values. Starting fresh.")
            return {}, None

        cached = {
            int(index): vectors[position]
            for position, index in enumerate(indices)
        }

        dimension = vectors.shape[1]

        print(
            f"Resuming embedding cache: "
            f"{len(cached)}/{len(documents)} documents embedded."
        )

        return cached, dimension

    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Could not load embedding cache: {exc}")
        return {}, None


def save_embedding_cache(
    documents,
    embeddings_by_index,
    dimension,
):
    """Atomically save completed document embeddings."""

    DB_DIR.mkdir(parents=True, exist_ok=True)

    indices = sorted(embeddings_by_index)

    if indices:
        vectors = np.vstack([
            embeddings_by_index[index]
            for index in indices
        ]).astype(np.float32)
    else:
        vectors = np.empty(
            (0, dimension or 0),
            dtype=np.float32,
        )

    temp_npz = EMBEDDING_CACHE_PATH.with_suffix(".tmp.npz")

    np.savez_compressed(
        temp_npz,
        indices=np.asarray(indices, dtype=np.int64),
        vectors=vectors,
    )

    os.replace(
        temp_npz,
        EMBEDDING_CACHE_PATH,
    )

    metadata = {
        "model": EMBEDDING_MODEL_NAME,
        "fingerprint": get_cache_fingerprint(documents),
        "num_cached": len(indices),
        "dimension": dimension,
    }

    temp_json = EMBEDDING_CACHE_META_PATH.with_suffix(".tmp")

    with open(
        temp_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    os.replace(
        temp_json,
        EMBEDDING_CACHE_META_PATH,
    )


# ============================================================
# Local SentenceTransformer embeddings
# ============================================================

def embed_batch(session, texts):
    """Backward-compatible helper for a single local embedding batch."""
    return local_embed_texts(texts, batch_size=len(texts))

    payload = {
        "model": EMBEDDING_MODEL_NAME,
        "input": texts,
        "encoding_format": "float",
    }

    response_data = post_embedding_request(
        session,
        payload,
    )

    items = response_data.get("data")

    if not isinstance(items, list):
        raise RuntimeError(
            f"Unexpected embedding API response: "
            f"{json.dumps(response_data)[:1000]}"
        )

    if len(items) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embeddings; "
            f"received {len(items)}."
        )

    # Preserve input order using provider-returned indices.
    if not all(
        isinstance(item, dict) and "index" in item
        for item in items
    ):
        raise RuntimeError(
            "Embedding response items are missing input indices."
        )

    items = sorted(
        items,
        key=lambda item: int(item["index"]),
    )

    vectors = []

    for item in items:
        vector = np.asarray(
            item.get("embedding"),
            dtype=np.float32,
        )

        if vector.ndim != 1:
            raise RuntimeError(
                f"Invalid embedding shape: {vector.shape}"
            )

        if not np.isfinite(vector).all():
            raise RuntimeError(
                "Received NaN or infinite embedding values."
            )

        norm = np.linalg.norm(vector)

        if norm == 0:
            raise RuntimeError(
                "Received a zero-magnitude embedding."
            )

        vector /= norm
        vectors.append(vector)

    dimensions = {len(vector) for vector in vectors}

    if len(dimensions) != 1:
        raise RuntimeError(
            f"Inconsistent embedding dimensions: {dimensions}"
        )

    return np.vstack(vectors).astype(np.float32)


def embed_texts(
    documents,
    batch_size=BATCH_SIZE,
):
    """
    Embed complete documents using SentenceTransformers.

    Completed batches are persisted, allowing interrupted indexing
    to resume without re-embedding successful documents.
    """

    if not documents:
        raise ValueError(
            "Cannot embed an empty document list."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    texts = [
        str(doc["full_text"])
        for doc in documents
    ]

    embeddings_by_index, dimension = load_embedding_cache(
        documents
    )

    missing_indices = [
        index
        for index in range(len(texts))
        if index not in embeddings_by_index
    ]

    print("Documents requiring embeddings:", len(missing_indices))
    print("Embedding batch size:", batch_size)
    print("Embedding provider: SentenceTransformers")

    if missing_indices:
        for batch_start in range(0, len(missing_indices), batch_size):
                batch_indices = missing_indices[batch_start:batch_start + batch_size]
                batch_vectors = local_embed_texts(
                    [texts[index] for index in batch_indices],
                    batch_size=batch_size,
                )

                current_dimension = batch_vectors.shape[1]

                if (
                    dimension is not None
                    and current_dimension != dimension
                ):
                    raise ValueError(
                        "Embedding dimension changed: "
                        f"expected {dimension}, "
                        f"received {current_dimension}."
                    )

                dimension = current_dimension

                for index, vector in zip(
                    batch_indices,
                    batch_vectors,
                ):
                    embeddings_by_index[index] = vector

                # Save immediately after every successful batch.
                save_embedding_cache(
                    documents,
                    embeddings_by_index,
                    dimension,
                )

                print(
                    f"Embedded "
                    f"{len(embeddings_by_index)}/{len(documents)} "
                    f"documents."
                )

    if len(embeddings_by_index) != len(documents):
        raise RuntimeError(
            "Embedding process incomplete: "
            f"{len(embeddings_by_index)}/{len(documents)} documents."
        )

    embeddings = np.vstack([
        embeddings_by_index[index]
        for index in range(len(documents))
    ]).astype(np.float32)

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Final embeddings contain NaN or infinite values."
        )

    if dimension is None or embeddings.shape[1] != dimension:
        raise ValueError(
            "Final embedding dimensions are inconsistent."
        )

    return np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )


# ============================================================
# Build dense complete-document index
# ============================================================

def build_dense_documents(documents):
    if not documents:
        raise ValueError(
            "Cannot build dense index: no documents available."
        )

    embeddings = embed_texts(
        documents,
        batch_size=BATCH_SIZE,
    )

    if len(embeddings) != len(documents):
        raise ValueError(
            "Embedding count does not match document count."
        )

    dense_index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    dense_index.add(embeddings)

    return dense_index


# ============================================================
# Save JSON
# ============================================================

def save_json(path, data):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Main
# ============================================================

def main():
    DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading corpus...")
    df = load_corpus()
    print("Corpus documents:", len(df))

    print("\nPreparing complete documents...")
    documents = create_documents(df)
    print("Complete documents:", len(documents))

    print("\nCreating BM25 chunks...")
    chunks = create_chunks(documents)
    print("Total chunks:", len(chunks))

    if not chunks:
        raise ValueError(
            "No non-empty chunks were generated."
        )

    print("\nBuilding BM25 index...")
    bm25_index, tokenized_corpus = build_bm25(chunks)

    print("\nBuilding local SentenceTransformers dense index...")
    dense_index = build_dense_documents(documents)

    print("\nSaving indexes...")

    with open(DB_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(
            {
                "bm25_index": bm25_index,
                "tokenized_corpus": tokenized_corpus,
            },
            f,
        )

    faiss.write_index(
        dense_index,
        str(DB_DIR / "dense_documents.faiss"),
    )

    save_json(
        DB_DIR / "chunks.json",
        chunks,
    )

    save_json(
        DB_DIR / "documents.json",
        documents,
    )

    config = {
        "dataset": DATASET_NAME,
        "embedding_backend": "sentence-transformers",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "dense_index_type": "complete_documents",
        "bm25_index_type": "chunks",
        "embedding_dimension": dense_index.d,
        "normalized_embeddings": True,
        "embedding_model_saved": False,
        "embedding_batch_size": BATCH_SIZE,
    }

    save_json(
        DB_DIR / "config.json",
        config,
    )

    print("\n" + "=" * 55)
    print("INDEX CONSTRUCTION COMPLETED")
    print("=" * 55)
    print("Documents:", len(documents))
    print("BM25 chunks:", len(chunks))
    print("Dense document vectors:", dense_index.ntotal)
    print("Embedding dimension:", dense_index.d)
    print("Embedding model:", EMBEDDING_MODEL_NAME)
    print("Database:", DB_DIR.resolve())
    print("Embedding cache:", EMBEDDING_CACHE_PATH)


if __name__ == "__main__":
    main()
