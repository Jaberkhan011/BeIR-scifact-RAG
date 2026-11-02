import os

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = os.getenv("DATASET_NAME", "scifact").lower()
DATA_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "datasets"))
DB_ROOT = Path(os.getenv("DB_ROOT", PROJECT_ROOT /
               "src/beir_scifact_rag/retrieval_db"))

DATASET_DIR = DATA_ROOT / DATASET_NAME
LEGACY_DATASET_DIR = DATA_ROOT
if DATASET_NAME == "scifact" and not DATASET_DIR.exists() and (LEGACY_DATASET_DIR / "corpus.parquet").exists():
    DATASET_DIR = LEGACY_DATASET_DIR

CORPUS_PATH = Path(os.getenv("CORPUS_PATH", DATASET_DIR / "corpus.parquet"))
QUESTIONS_PATH = Path(
    os.getenv("QUESTIONS_PATH", DATASET_DIR / "questions.parquet"))
QRELS_PATH = Path(os.getenv("QRELS_PATH", DATASET_DIR / "qrels.parquet"))

SCOPED_DB_DIR = DB_ROOT / DATASET_NAME
if DATASET_NAME == "scifact" and not SCOPED_DB_DIR.exists() and (DB_ROOT / "config.json").exists():
    SCOPED_DB_DIR = DB_ROOT
DB_DIR = Path(os.getenv("DB_DIR", SCOPED_DB_DIR))

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100


EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
)

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

# Retrieval
TOP_K = 10
CANDIDATE_K = 100
# Reranking is intentionally applied to a smaller hybrid shortlist.
RERANK_CANDIDATE_K = min(
    50,
    max(20, int(os.getenv("RERANK_CANDIDATE_K", "50"))),
)

# Standard RRF parameter
RRF_K = 60
