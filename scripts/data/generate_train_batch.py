#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:10:31 2026

@author: tbaker
"""

import argparse
from pathlib import Path


def main():
    """Parse command-line arguments and submit a training-data batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to the CSV, JSON, or JSONL source dataset.",
    )
    parser.add_argument(
        "--templates",
        "--template",
        required=True,
        nargs="+",
        type=Path,
        help=(
            "Template paths in application order. Each filename stem names "
            "its generated field. The last field is submitted as the prompt."
        ),
    )
    parser.add_argument(
        "--framework",
        type=Path,
        help="Text file to supply as the optional {framework} variable.",
    )
    parser.add_argument(
        "--model",
        default="gpt-6-luna",
        help="OpenAI model for the batch (default: %(default)s).",
    )
    parser.add_argument(
        "--tag",
        help="Label used to name the retrieved dataset.",
    )
    parser.add_argument(
        "--description",
        help="Description to attach to the batch and save in its metadata.",
    )
    parser.add_argument(
        "--metadata_dir",
        type=Path,
        default=Path("outputs/batches"),
        help="Directory for batch metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Print up to three batch JSONL lines without submitting.",
    )
    args = parser.parse_args()

    # Preserve template order so later templates can use earlier fields.
    field_templates = {path.stem: path for path in args.templates}
    if len(field_templates) != len(args.templates):
        parser.error("Template filenames must have distinct stems.")
    prompt_field = args.templates[-1].stem

    from llm_research.data.generation import submit_batch_completions

    # Pass framework text as a variable so it is inserted verbatim.
    template_vars = {}
    if args.framework is not None:
        template_vars["framework"] = args.framework.read_text(encoding="utf-8")

    batch = submit_batch_completions(
        args.dataset,
        field_templates,
        prompt_field,
        args.model,
        metadata_dir=args.metadata_dir,
        tag=args.tag,
        description=args.description,
        template_vars=template_vars,
        test_mode=args.test_mode,
    )
    if batch is not None:
        print(f"Submitted batch {batch.id}")


if __name__ == "__main__":
    main()
