#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 22:34:42 2026

@author: tbaker
"""
import argparse


DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
DEFAULT_ADAPTER = None
DEFAULT_PROMPTS = "data/toy_eval_prompts.jsonl"
DEFAULT_OUTPUT = "outputs/results/toy_eval.jsonl"

# Colors for console output!
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"


def print_result(result, index, total):
    print(f"{GREEN}[{index}/{total}]{RESET}")
    print(result["prompt"], end="", flush=True)
    print(f"{YELLOW}{result['response']}{RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a set of prompts through an LLM and save the outputs."
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Base model name/path (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        help="Path to a LoRA adapter. Omit to use the base model.",
    )

    parser.add_argument(
        "--prompts",
        default=DEFAULT_PROMPTS,
        help=f"Prompt JSONL file (default: {DEFAULT_PROMPTS})",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args()

    from llm_research.evaluation import run_evaluation

    run_evaluation(
        model_name=args.model,
        adapter_path=args.adapter,
        prompts_path=args.prompts,
        output_path=args.output,
        callbacks=[print_result],
    )
    


if __name__ == "__main__":
    main()

