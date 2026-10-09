#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:10:31 2026

@author: tbaker
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()
    
    
    from llm_research.data_generation import generate_training_data

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open output filepath
    with output_path.open("w", encoding="utf-8") as f:
        # Generate the training data and iterate through results
        for item in generate_training_data(
                args.dataset,
                args.template,
                args.framework,
                ):
            
            # Make a scenario/response pair, and write to output file
            record = {
                "scenario": item["scenario"],
                "response": item["response"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()