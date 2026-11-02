import json
from beir_scifact_rag.retrieval.pipeline import (
    bm25_search,
    dense_search,
    hybrid_search,
    hybrid_rerank_search,
)
from beir_scifact_rag.config import (
    DATASET_NAME,
    QUESTIONS_PATH,
    QRELS_PATH,
    TOP_K,
    CANDIDATE_K,
    RERANK_CANDIDATE_K,
)
import pytrec_eval
import pandas as pd
from collections import defaultdict
import os
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ============================================================
# Configuration
# ============================================================

EVALUATION_ROOT = Path(os.getenv("EVALUATION_ROOT", "evaluation_results"))
DATASET_OUTPUT_DIR = EVALUATION_ROOT / DATASET_NAME
DATASET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = DATASET_OUTPUT_DIR / "aggregate.csv"
PER_QUERY_OUTPUT_PATH = DATASET_OUTPUT_DIR / "per_query.csv"
RUN_OUTPUT_PATH = DATASET_OUTPUT_DIR / "runs.json"

# Focus the benchmark on multi-evidence questions: queries with at least two
# positively judged corpus documents require retrieving more than one chunk.
MIN_RELEVANT_DOCS = int(os.getenv("MIN_RELEVANT_DOCS", "2"))
EVAL_QUERY_LIMIT = int(os.getenv("EVAL_QUERY_LIMIT", "50"))


# ============================================================
# Load queries
# ============================================================

questions_df = pd.read_parquet(QUESTIONS_PATH)
qrels_df = pd.read_parquet(QRELS_PATH)

required_question_columns = {"_id", "title", "text"}
required_qrels_columns = {"query-id", "corpus-id", "score"}

missing_questions = required_question_columns - set(
    questions_df.columns
)
missing_qrels = required_qrels_columns - set(
    qrels_df.columns
)

if missing_questions:
    raise ValueError(
        f"Missing columns in questions parquet: {missing_questions}"
    )

if missing_qrels:
    raise ValueError(
        f"Missing columns in qrels parquet: {missing_qrels}"
    )

# Normalize identifiers to strings.
questions_df["_id"] = questions_df["_id"].astype(str)

qrels_df["query-id"] = qrels_df["query-id"].astype(str)
qrels_df["corpus-id"] = qrels_df["corpus-id"].astype(str)

# Clean query text.
questions_df["title"] = (
    questions_df["title"].fillna("").astype(str)
)

questions_df["text"] = (
    questions_df["text"].fillna("").astype(str)
)

questions_df["query"] = (
    questions_df["title"].str.strip()
    + "\n"
    + questions_df["text"].str.strip()
).str.strip()

# Remove empty queries and duplicate query IDs.
questions_df = questions_df[
    questions_df["query"].str.len() > 0
].drop_duplicates(subset="_id", keep="first")

queries = dict(
    zip(
        questions_df["_id"],
        questions_df["query"],
    )
)


# ============================================================
# Build qrels
# ============================================================

qrels = defaultdict(dict)

for row in qrels_df.itertuples(index=False):
    qid = str(getattr(row, "_0", ""))  # Not used; see explicit columns below.

# Access the columns directly to avoid dependence on pandas
# namedtuple field renaming for hyphenated column names.
qrels.clear()

for qid, doc_id, score in zip(
    qrels_df["query-id"],
    qrels_df["corpus-id"],
    qrels_df["score"],
):
    if pd.isna(qid) or pd.isna(doc_id) or pd.isna(score):
        continue

    qid = str(qid)
    doc_id = str(doc_id)

    # SciFact relevance labels are integer grades.
    relevance = int(float(score))

    qrels[qid][doc_id] = relevance

qrels = dict(qrels)

# Evaluate only queries that have both query text and qrels.
valid_qids = sorted(
    qid
    for qid, relevant_docs in qrels.items()
    if qid in queries
    and sum(score > 0 for score in relevant_docs.values()) >= MIN_RELEVANT_DOCS
)

if EVAL_QUERY_LIMIT > 0:
    valid_qids = valid_qids[:EVAL_QUERY_LIMIT]

if not valid_qids:
    raise ValueError(
        "No evaluable queries found. Check query IDs and qrels IDs."
    )

print("=" * 70)
print("EVALUATION CONFIGURATION")
print("=" * 70)
print("Queries loaded:", len(queries))
print("Qrels query IDs:", len(qrels))
print("Evaluable queries:", len(valid_qids))
print("Minimum relevant documents/query:", MIN_RELEVANT_DOCS)
print("TOP_K:", TOP_K)
print("CANDIDATE_K:", CANDIDATE_K)
print("Reranking candidate count:", RERANK_CANDIDATE_K)


# ============================================================
# Retrieval functions
# ============================================================

def unique_document_ranking(results, k=TOP_K):
    """
    Convert a list of chunk/document results into a ranking
    of unique original document IDs.

    The first occurrence determines the rank.
    """

    ranked_docs = []
    seen = set()

    for result in results:
        doc_id = str(result["doc_id"])

        if doc_id in seen:
            continue

        seen.add(doc_id)
        ranked_docs.append(doc_id)

        if len(ranked_docs) >= k:
            break

    return ranked_docs


def retrieve_bm25(query, k=TOP_K):
    """
    BM25 searches chunks. Retrieve a larger candidate set,
    then deduplicate by original document ID.
    """

    results = bm25_search(
        query,
        k=max(CANDIDATE_K, k),
    )

    return unique_document_ranking(results, k=k)


def retrieve_dense(query, k=TOP_K):
    """
    Dense search returns complete documents, so the result
    list is already document-level.
    """

    results = dense_search(
        query,
        k=max(CANDIDATE_K, k),
    )

    return unique_document_ranking(results, k=k)


def retrieve_hybrid(query, k=TOP_K):
    """Evaluate document-level Reciprocal Rank Fusion."""

    results = hybrid_search(
        query,
        k=max(CANDIDATE_K, k),
        candidate_k=max(CANDIDATE_K, k),
    )

    return unique_document_ranking(results, k=k)


def retrieve_crossencoder_reranked(query, k=TOP_K):
    """Evaluate RRF followed by the local CrossEncoder."""

    results = hybrid_rerank_search(
        query,
        k=k,
        candidate_k=RERANK_CANDIDATE_K,
        reranker_backend="crossencoder",
    )

    return unique_document_ranking(results, k=k)


# ============================================================
# Metric evaluator
# ============================================================

def evaluate_system(name, retriever, k=TOP_K):
    """
    Evaluate a retrieval system on the same fixed set of queries.

    Returns:
        aggregate_metrics: macro-averaged metrics
        per_query_metrics: one row per query
        run: TREC-style document ranking run
    """

    run = {}
    elapsed_seconds = 0.0
    failed_queries = []
    query_times = {}

    for position, qid in enumerate(valid_qids, start=1):
        query = queries[qid]

        start_time = time.perf_counter()

        try:
            ranked_docs = retriever(query, k=k)
        except Exception as exc:
            print(
                f"\n[{name}] Failed on query {qid}: "
                f"{type(exc).__name__}: {exc}"
            )
            failed_queries.append(qid)
            ranked_docs = []

        query_time_seconds = time.perf_counter() - start_time
        elapsed_seconds += query_time_seconds
        query_times[qid] = query_time_seconds

        # Defensive document-ID deduplication.
        ranked_docs = list(
            dict.fromkeys(str(doc_id) for doc_id in ranked_docs)
        )[:k]

        # pytrec_eval expects document scores, with larger scores
        # corresponding to higher-ranked documents.
        run[qid] = {
            doc_id: float(k - rank + 1)
            for rank, doc_id in enumerate(ranked_docs, start=1)
        }

        if position % 50 == 0 or position == len(valid_qids):
            print(
                f"\r{name}: {position}/{len(valid_qids)} queries",
                end="",
                flush=True,
            )

    print()

    metric_names = {
        f"P_{k}", f"recall_{k}", f"map_cut_{k}",
        "recip_rank", f"ndcg_cut_{k}", f"success_{k}",
    }

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        metric_names,
    )

    per_query_scores = evaluator.evaluate(run)

    # Keep all evaluated query IDs, including failed/empty runs.
    # This ensures that failures do not silently disappear
    # from the macro-average.
    per_query_rows = []

    for qid in valid_qids:
        scores = per_query_scores.get(qid, {})

        per_query_rows.append({
            "system": name,
            "query_id": qid,
            "num_relevant": sum(score > 0 for score in qrels[qid].values()),
            f"Precision@{k}": scores.get(f"P_{k}", 0.0),
            f"Recall@{k}": scores.get(f"recall_{k}", 0.0),
            f"MAP@{k}": scores.get(f"map_cut_{k}", 0.0),
            "MRR": scores.get("recip_rank", 0.0),
            f"NDCG@{k}": scores.get(f"ndcg_cut_{k}", 0.0),
            f"Success@{k}": scores.get(f"success_{k}", 0.0),
            "retrieval_failed": qid in failed_queries,
            "query_time_s": query_times[qid],
        })

    per_query_df = pd.DataFrame(per_query_rows)

    # Macro-average across the same query set.
    aggregate_metrics = {
        f"Precision@{k}": per_query_df[f"Precision@{k}"].mean(),
        f"Recall@{k}": per_query_df[f"Recall@{k}"].mean(),
        f"MAP@{k}": per_query_df[f"MAP@{k}"].mean(),
        "MRR": per_query_df["MRR"].mean(),
        f"NDCG@{k}": per_query_df[f"NDCG@{k}"].mean(),
        f"Success@{k}": per_query_df[f"Success@{k}"].mean(),
        "Avg query time (s)": elapsed_seconds / len(valid_qids),
        "P50 query time (s)": per_query_df["query_time_s"].median(),
        "P95 query time (s)": per_query_df["query_time_s"].quantile(0.95),
        "Queries": len(valid_qids),
        "Failed queries": len(failed_queries),
    }

    return aggregate_metrics, per_query_df, run


# ============================================================
# Configure benchmark systems
# ============================================================

systems = {
    "BM25": retrieve_bm25,
    "Dense": retrieve_dense,
    "Hybrid RRF": retrieve_hybrid,
}

systems["Hybrid + Reranker"] = retrieve_crossencoder_reranked


# ============================================================
# Run benchmark
# ============================================================

aggregate_results = {}
all_per_query_results = []
all_runs = {}

for name, retriever in systems.items():
    print("\n" + "=" * 70)
    print("Evaluating:", name)
    print("=" * 70)

    metrics, per_query_df, run = evaluate_system(
        name=name,
        retriever=retriever,
        k=TOP_K,
    )

    aggregate_results[name] = metrics
    all_per_query_results.append(per_query_df)
    all_runs[name] = run

    print(
        pd.Series(metrics).to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )


# ============================================================
# Display aggregate results
# ============================================================

results_df = pd.DataFrame.from_dict(
    aggregate_results,
    orient="index",
)

metric_columns = [
    f"Precision@{TOP_K}",
    f"Recall@{TOP_K}",
    f"MAP@{TOP_K}",
    "MRR",
    f"NDCG@{TOP_K}",
    f"Success@{TOP_K}",
    "Avg query time (s)",
    "P50 query time (s)",
    "P95 query time (s)",
    "Queries",
    "Failed queries",
]

results_df = results_df[metric_columns]

print("\n" + "=" * 70)
print("FINAL RETRIEVAL RESULTS")
print("=" * 70)
print(results_df.round(4).to_string())


# ============================================================
# Save aggregate and per-query metrics
# ============================================================

results_df.to_csv(OUTPUT_PATH, index=True)

per_query_results_df = pd.concat(
    all_per_query_results,
    ignore_index=True,
)

per_query_results_df.to_csv(
    PER_QUERY_OUTPUT_PATH,
    index=False,
)


with open(RUN_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_runs, f, indent=2)

print(f"\nDataset: {DATASET_NAME}")
print(f"Saved aggregate metrics to: {OUTPUT_PATH}")
print(f"Saved per-query metrics to: {PER_QUERY_OUTPUT_PATH}")
print(f"Saved runs to: {RUN_OUTPUT_PATH}")
