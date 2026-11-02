"""Download one or more BEIR datasets into dataset-specific directories."""

import argparse
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


def download_dataset(dataset_name: str, output_root: Path) -> Path:
    name = dataset_name.lower().strip()
    dataset_dir = output_root / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels = load_dataset(f"BeIR/{name}-qrels", split="test")

    paths = {
        "corpus": dataset_dir / "corpus.parquet",
        "questions": dataset_dir / "questions.parquet",
        "qrels": dataset_dir / "qrels.parquet",
    }
    corpus.to_parquet(paths["corpus"])
    queries.to_parquet(paths["questions"])
    qrels.to_parquet(paths["qrels"])

    print(f"{name}: corpus={corpus.shape}, queries={queries.shape}, qrels={qrels.shape}")
    print(f"Saved to {dataset_dir}")
    return dataset_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact"],
        help="BEIR dataset names, for example scifact nfcorpus fiqa.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("datasets"))
    args = parser.parse_args()

    for dataset_name in args.datasets:
        download_dataset(dataset_name, args.output_root)


if __name__ == "__main__":
    main()
