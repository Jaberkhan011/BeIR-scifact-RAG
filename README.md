# BeIR SciFact RAG

An end-to-end hybrid retrieval benchmark for the **SciFact** dataset. The
pipeline combines BM25 over text chunks, normalized SentenceTransformers
embeddings, reciprocal-rank fusion, and local cross-encoder reranking.

The repository contains source code only. Downloaded datasets, generated
indexes, model caches, and evaluation output stay local and are ignored by
Git.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Enough disk space for the SciFact data and locally cached models

## Setup

```bash
uv sync
cp .env.example .env
```

The defaults use `datasets/scifact/` for data and
`src/beir_scifact_rag/retrieval_db/scifact/` for the generated index. Edit
`.env` only when you need different paths or model settings. Never commit
`.env` or real API keys.

## Download SciFact

```bash
uv run python -m beir_scifact_rag.data.download --datasets scifact
```

This writes `corpus.parquet`, `questions.parquet`, and `qrels.parquet` to
`datasets/scifact/`.

## Build the retrieval index

```bash
uv run python -m beir_scifact_rag.indexing.build
```

The index contains BM25 data, chunk metadata, document metadata, and a FAISS
dense index. All generated files are local and ignored by Git.

## Run evaluation

```bash
uv run python -m beir_scifact_rag.evaluation.run
```

Results are written to `evaluation_results/scifact/`. The SciFact benchmark
outputs are included in the repository for reproducibility. By default,
evaluation uses queries with at least two relevant documents and evaluates up to 50
queries. To evaluate all matching queries:

```bash
EVAL_QUERY_LIMIT=0 uv run python -m beir_scifact_rag.evaluation.run
```

Reported metrics include Precision@10, Recall@10, Recall@100, MAP@10,
MAP@100, MRR@10, NDCG@10, NDCG@100, Hit Rate@10, latency, and failed-query
count.

The published benchmark files are:

- `evaluation_results/scifact/aggregate.csv` — aggregate metrics by retriever
- `evaluation_results/scifact/per_query.csv` — per-query metrics
- `evaluation_results/scifact/runs.json` — saved ranked results

## Source layout

```text
src/beir_scifact_rag/
├── data/          SciFact download utilities
├── indexing/      BM25 and dense index construction
├── retrieval/     Hybrid retrieval and reranking
├── evaluation/    Benchmark runner
└── config.py      Environment and model configuration
```

## Git hygiene

The following local artifacts are intentionally excluded from commits:

- `.env` and secret files
- `datasets/`
- `src/beir_scifact_rag/retrieval_db/`
- generated evaluation output except the published SciFact benchmark in
  `evaluation_results/scifact/`
- local notes and OS metadata

To inspect what would be committed before pushing:

```bash
git status --short
git check-ignore -v datasets/scifact/corpus.parquet \
  src/beir_scifact_rag/retrieval_db/scifact/dense_documents.faiss \
  .env
```
