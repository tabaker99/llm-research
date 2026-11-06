#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:10:31 2026

@author: tbaker
"""

import argparse
import json
from pathlib import Path



# Colors for console output!
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the CSV dataset containing moral scenarios and actions.",
    )
    
    parser.add_argument(
        "--template",
        required=True,
        help="Path to the prompt template used to generate model responses.",
    )
    
    parser.add_argument(
        "--framework",
        required=True,
        help="Path to the text file containing the moral framework.",
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSONL training-data file.",
    )
    
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Enable test mode for a small trial run.",
    )

    args = parser.parse_args()
    
    
    from llm_research.data_generation import (generate_training_data,
                                              construct_scenarios,
                                              enable_testing_mode)
    
    if args.test_mode:
        enable_testing_mode()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open output filepath
    with output_path.open("w", encoding="utf-8") as f:
        scenarios, n = construct_scenarios(args.dataset)
        # Generate the training data and iterate through results
        for i, item in enumerate(
                generate_training_data(
                    scenarios,
                    args.framework,
                    args.template,
                )):
            print(f"{GREEN}{i+1}/{n}{RESET}")
            scenario = item["scenario"]
            response = item["response"]
            print(f"SCENARIO:\n\n{scenario}\n\n")
            print(f"RESPONSE:\n\n{response}\n\n")
            
            # Make a scenario/response pair, and write to output file
            record = {
                "scenario": scenario,
                "response": response,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()