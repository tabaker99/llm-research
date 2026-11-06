#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:38:15 2026

@author: tbaker
"""

import argparse
import csv
import random


def shuffle_dataset_columns(input_path, output_path, columns, seed=42):
    """
    Shuffle selected column values within each row of a CSV dataset.
    
    The values in the specified columns are randomly permuted independently
    for each row. All other columns are left unchanged. A fixed random seed
    may be used to make the shuffling reproducible.
    
    Parameters
    ----------
    input_path : str or path-like
        Path to the input CSV file.
    output_path : str or path-like
        Path where the shuffled CSV file will be written.
    columns : sequence of str
        Names of the columns whose values should be shuffled within each row.
    seed : int, optional
        Random seed used for reproducible shuffling. The default is 42.
    
    Raises
    ------
    ValueError
        If the input file has no header or if any requested column is not
        present in the dataset.
    """
    
    rng = random.Random(seed)

    with open(input_path, "r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)

        if reader.fieldnames is None:
            raise ValueError("Input file has no header.")

        missing = [column for column in columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Columns not found: {missing}")

        rows = list(reader)

    for row in rows:
        values = [row[column] for column in columns]
        rng.shuffle(values)

        for column, value in zip(columns, values):
            row[column] = value

    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Randomly shuffle selected columns within each dataset row."
    )

    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Column names whose values should be shuffled within each row.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()

    shuffle_dataset_columns(
        args.input,
        args.output,
        args.columns,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()