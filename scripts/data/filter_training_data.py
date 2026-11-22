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

    parser.add_argument("file_a", help="First JSONL dataset.")
    parser.add_argument("file_b", help="Second JSONL dataset.")
    parser.add_argument(
        "--response_field",
        type=str,
        default="completion",
        help=("Name of the field that contains a response with REASONING "
              "and ACTION tags.")
    )
    
    parser.add_argument(
        "--id_field",
        type=str,
        default="scenario_id",
        help=("Name of the field that contains a unique scenario id.")
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing output files.",
    )

    args = parser.parse_args()

    def condition_function(reasoning_a, reasoning_b):
        if ("framework" in reasoning_a.lower()
            or "framework" in reasoning_b.lower()):
            return False
        return True

    results = filter_matching_actions(
        args.file_a,
        args.file_b,
        args.overwrite,
        response_field=args.response_field,
        id_field=args.id_field,
        condition_function=condition_function
    )

    print(f"Total scenarios: {results['total']}")
    print(f"Matching actions: {results['matched']}")
    print(f"Disagreements: {results['disagreed']}")
    print(f"Broken condition: {results['broken_condition']}")
    print(f"Invalid actions: {results['invalid']}")
    print(f"Saved: {results['output_a']}")
    print(f"Saved: {results['output_b']}")


if __name__ == "__main__":
    main()