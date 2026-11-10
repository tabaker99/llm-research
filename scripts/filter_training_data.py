#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:25:43 2026

@author: tbaker
"""


"""Filter paired JSONL datasets by action agreement."""

import argparse

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

    args = parser.parse_args()

    results = filter_matching_actions(args.file_a, args.file_b)

    print(f"Total scenarios: {results['total']}")
    print(f"Matching actions: {results['matched']}")
    print(f"Disagreements: {results['disagreed']}")
    print(f"Invalid actions: {results['invalid']}")
    print(f"Saved: {results['output_a']}")
    print(f"Saved: {results['output_b']}")


if __name__ == "__main__":
    main()