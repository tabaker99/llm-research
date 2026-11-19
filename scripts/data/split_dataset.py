#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:49:10 2026

@author: tbaker
"""


"""Split two aligned JSONL datasets into training and evaluation sets."""

import argparse
import json
import random
from pathlib import Path


def split_datasets(path_a, path_b, test_size=0.2, seed=42):
    """Split paired JSONL files using identical scenario indices."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if Path(path_a).resolve() == Path(path_b).resolve():
        raise ValueError("Input files must be different")

    with open(path_a, "r", encoding="utf-8") as file:
        lines_a = file.readlines()

    with open(path_b, "r", encoding="utf-8") as file:
        lines_b = file.readlines()

    if len(lines_a) != len(lines_b):
        raise ValueError("Input files have different lengths")

    for i, (line_a, line_b) in enumerate(zip(lines_a, lines_b)):
        if json.loads(line_a)["scenario"] != json.loads(line_b)["scenario"]:
            raise ValueError(f"Scenario mismatch at line {i + 1}")

    indices = list(range(len(lines_a)))
    random.Random(seed).shuffle(indices)

    n_test = round(len(indices) * test_size)
    test_indices = set(indices[:n_test])

    for path, lines in ((path_a, lines_a), (path_b, lines_b)):
        path = Path(path)

        train_path = path.with_name(f"{path.stem}.train.jsonl")
        eval_path = path.with_name(f"{path.stem}.eval.jsonl")

        for output in (train_path, eval_path):
            if output.exists():
                raise FileExistsError(f"Output already exists: {output}")

        with (
            open(train_path, "w", encoding="utf-8", newline="") as train,
            open(eval_path, "w", encoding="utf-8", newline="") as eval_file,
        ):
            for i, line in enumerate(lines):
                if i in test_indices:
                    eval_file.write(line)
                else:
                    train.write(line)

    print(f"Training scenarios: {len(indices) - n_test}")
    print(f"Evaluation scenarios: {n_test}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split two aligned JSONL datasets into matching "
            "training and evaluation sets. Results will be saved to files "
            "with `.train` and `.eval` added before the file extension."
        )
    )

    parser.add_argument("file_a", help="First JSONL dataset")
    parser.add_argument("file_b", help="Second JSONL dataset")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    split_datasets(
        args.file_a,
        args.file_b,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()