#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 19:32:53 2026

@author: tbaker
"""


import argparse
import csv
import json
from pathlib import Path



def merge_csv_fields(
    jsonl_path,
    csv_path,
    output_path,
    ignore_fields=None,
    csv_fields_first=False,
):
    """
    Add missing CSV fields to a JSONL file, matching rows by position.

    Parameters
    ----------
    jsonl_path : str or Path
        Input JSONL file.
    csv_path : str or Path
        Input CSV file.
    output_path : str or Path
        Destination JSONL file.
    ignore_fields : iterable of str, optional
        CSV fields to exclude from merging.
    csv_fields_first : bool, default=False
        If True, place CSV fields before JSONL-only fields, preserving
        the original order of each source.

    Raises
    ------
    ValueError
        If the files have different numbers of rows.
    """
    ignore_fields = set(ignore_fields or [])

    # Load JSONL data.
    with open(jsonl_path, "r", encoding="utf-8") as f:
        jsonl_data = [json.loads(line) for line in f if line.strip()]

    # Load CSV data.
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        csv_fields = reader.fieldnames or []
        csv_data = list(reader)

    if len(jsonl_data) != len(csv_data):
        raise ValueError(
            f"Row count mismatch: JSONL has {len(jsonl_data)} rows, "
            f"CSV has {len(csv_data)} rows."
        )

    # Determine which CSV fields to import.
    included_fields = [
        field for field in csv_fields
        if field not in ignore_fields
    ]

    # Track field classifications.
    jsonl_fields = set().union(*(row.keys() for row in jsonl_data))
    added = set(included_fields) - jsonl_fields
    already_present = set(included_fields) & jsonl_fields
    skipped = set(csv_fields) & ignore_fields

    # Merge rows.
    merged_data = []

    for jsonl_row, csv_row in zip(jsonl_data, csv_data):
        csv_values = {
            field: csv_row[field]
            for field in included_fields
        }

        if csv_fields_first:
            merged = csv_values | jsonl_row
        else:
            merged = jsonl_row | {
                field: value
                for field, value in csv_values.items()
                if field not in jsonl_row
            }

        merged_data.append(merged)

    # Save result.
    with open(output_path, "w", encoding="utf-8") as f:
        for row in merged_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Print summary.
    print(f"\nMerged {len(merged_data)} rows.")
    print(f"Output: {output_path}")

    for label, fields in [
        ("Added", added),
        ("Skipped (ignored)", skipped),
        ("Already present", already_present),
    ]:
        print(f"\n{label} ({len(fields)}):")
        for field in sorted(fields):
            print(f"  - {field}")


def main():
    parser = argparse.ArgumentParser(
        description="Add missing CSV fields to a JSONL file."
    )

    parser.add_argument("jsonl_path")
    parser.add_argument("csv_path")
    parser.add_argument("output_path")
    parser.add_argument(
        "--ignore-fields",
        nargs="*",
        default=[],
        help="CSV fields to exclude from merging.",
    )
    
    parser.add_argument(
        "--csv-first",
        action="store_true",
        help="Order the csv fields before the jsonl fields.",
    )

    args = parser.parse_args()

    merge_csv_fields(
        args.jsonl_path,
        args.csv_path,
        args.output_path,
        ignore_fields=args.ignore_fields,
        csv_fields_first=args.csv_first,
    )


if __name__ == "__main__":
    main()