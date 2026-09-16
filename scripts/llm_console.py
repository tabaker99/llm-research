#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:04:47 2026

@author: tbaker
"""

from IPython import embed

from llm_research.llm_engine import LLMEngine


DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"


def main():
    print(f"Loading {DEFAULT_MODEL}...")

    engine = LLMEngine(DEFAULT_MODEL)

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