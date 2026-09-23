#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:04:47 2026

@author: tbaker
"""

import argparse
from IPython import embed

print("Starting LLM model...")

DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
DEFAULT_ADAPTER = "outputs/adapters/toy"  # change to your actual adapter folder


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        help=f"Path to a LoRA adapter (default: {DEFAULT_ADAPTER})",
    )

    args = parser.parse_args()
    
    from llm_research.engine import LLMEngine

    print(f"Loading {DEFAULT_MODEL}...")

    if args.adapter:
        print(f"Loading adapter: {args.adapter}")

    engine = LLMEngine(
        DEFAULT_MODEL,
        adapter_path=args.adapter,
    )

    print("\nLLM engine ready.")
    print("Access it with the variable: engine")
    print("Example: engine.complete('The capital of France is')\n")

    embed(
        header="Interactive LLM console",
        user_ns={"engine": engine},
        colors="linux",
    )


if __name__ == "__main__":
    main()