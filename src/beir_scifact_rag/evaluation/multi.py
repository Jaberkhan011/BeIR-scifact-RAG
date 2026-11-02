"""Run the retrieval benchmark independently for multiple BEIR datasets."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=os.getenv("DATASETS", "scifact").split(","),
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path(os.getenv("EVALUATION_ROOT", "evaluation_results")),
    )
    args = parser.parse_args()

    summaries = []
    script = Path(__file__).with_name("run.py")

    for raw_name in args.datasets:
        dataset_name = raw_name.strip().lower()
        if not dataset_name:
            continue

        env = os.environ.copy()
        env["DATASET_NAME"] = dataset_name
        env["EVALUATION_ROOT"] = str(args.evaluation_root)

        subprocess.run(
            [sys.executable, str(script)],
            env=env,
            check=True,
        )

        aggregate_path = args.evaluation_root / dataset_name / "aggregate.csv"
        aggregate = pd.read_csv(aggregate_path, index_col=0)
        aggregate.insert(0, "dataset", dataset_name)
        aggregate.insert(1, "system", aggregate.index)
        summaries.append(aggregate.reset_index(drop=True))

    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        args.evaluation_root.mkdir(parents=True, exist_ok=True)
        output_path = args.evaluation_root / "all_datasets.csv"
        combined.to_csv(output_path, index=False)
        print(f"Saved combined dataset summary to: {output_path}")


if __name__ == "__main__":
    main()
