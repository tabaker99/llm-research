#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:25:43 2026

@author: tbaker
"""


"""Filter paired JSONL datasets by action agreement."""

import argparse

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO
)
logger.setLevel(logging.WARNING)

from llm_research.data.filtering import filter_matching_actions


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter two aligned JSONL datasets, retaining examples "
            "where both moral systems choose the same valid action. "
            "For each input file, an output file will be created with the "
            "same name, but ending in `.filtered.jsonl`."
        )
    )

    parser.add_argument("file_a", help="First JSONL dataset")
    parser.add_argument("file_b", help="Second JSONL dataset")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing output files.",
    )

    args = parser.parse_args()

    results = filter_matching_actions(args.file_a, args.file_b, args.overwrite)

    print(f"Total scenarios: {results['total']}")
    print(f"Matching actions: {results['matched']}")
    print(f"Disagreements: {results['disagreed']}")
    print(f"Invalid actions: {results['invalid']}")
    print(f"Saved: {results['output_a']}")
    print(f"Saved: {results['output_b']}")


if __name__ == "__main__":
    main()