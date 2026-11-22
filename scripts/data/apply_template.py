#!/usr/bin/env python3
"""Render a template for each dataset row and save the result as JSONL."""

import argparse
import json
from pathlib import Path


def main():
    """Parse arguments, apply the template, and write the resulting rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to the source CSV, JSON, or JSONL dataset.",
    )
    parser.add_argument(
        "--template",
        required=True,
        type=Path,
        help="Path to a UTF-8 template with {column_name} placeholders.",
    )
    parser.add_argument(
        "--output-field",
        required=True,
        help="Field to add or replace with the rendered template.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to a new JSONL file (default: <dataset>.templated.jsonl).",
    )
    args = parser.parse_args()

    builders = {".csv": "csv", ".json": "json", ".jsonl": "json"}
    builder = builders.get(args.dataset.suffix.lower())
    if builder is None:
        parser.error("--dataset must be a CSV, JSON, or JSONL file.")

    output_path = args.output or args.dataset.with_name(
        f"{args.dataset.stem}.templated.jsonl"
    )

    from datasets import load_dataset

    from llm_research.data.processing import apply_templates

    try:
        template = args.template.read_text(encoding="utf-8")
        dataset = load_dataset(
            builder, data_files=str(args.dataset), split="train"
        )
        # The shared helper keeps rendered values as text even when the
        # output field previously held numbers.
        dataset = apply_templates(dataset, {args.output_field: template})

        # Open exclusively so the source and any existing exports are safe.
        output = output_path.open("x", encoding="utf-8")
        try:
            with output:
                for row in dataset:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
        except BaseException:
            # Allow a failed export to be retried at the same path.
            output_path.unlink(missing_ok=True)
            raise
    except KeyError as error:
        parser.error(f"Template field not found: {error}")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Saved {len(dataset)} rows to {output_path}")


if __name__ == "__main__":
    main()
