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

SCENARIO_TEMPLATE = ("CONTEXT:\n{context}\n\n"
                     "ACTIONS:\nA. {action1}\nB. {action2}")

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
    
    from datasets import load_dataset
    from llm_research.data_generation import generate_completions
    from llm_research.data.formatting import apply_template
    from llm_research.engine.base import TestCompletionEngine
    from llm_research.engine.openai import OpenAICompletionEngine
    
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(
        "csv",
        data_files=args.dataset,
        split="train",
    )
    
    
    # Build scenarios
    dataset = apply_template(
        dataset,
        "scenario",
        template=SCENARIO_TEMPLATE,
        remove_source_columns=True)
    
    # Load engine
    engine = TestCompletionEngine()
    
    # Load framework
    framework = Path(args.framework).read_text(encoding="utf-8")
    
    def format_info(row, prompt, completion):
        scenario_id = row["scenario_id"]
        scenario = row["scenario"]
        return (f"{GREEN}ID: {RESET}{scenario_id}\n"
                f"{GREEN}SCENARIO: \n{RESET}{scenario}\n"
                f"{GREEN}COMPLETION: \n{RESET}{completion}\n")
    
    # Generate completions and add to dataset
    completion_dataset = generate_completions(
        dataset,
        engine,
        log_function=format_info,
        template_path=args.template,
        template_vars={"framework": framework},
        )
    
    # Save new dataset to file
    completion_dataset.to_json(output_path, orient="records", lines=True)
    


if __name__ == "__main__":
    main()