#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 19:23:51 2026

@author: tbaker
"""

import argparse

from llm_research.config import load_config


DEFAULT_CONFIG = "./configs/DEFAULT.toml"


# Colors for console output!
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"


def print_data(result, index, total):
    prompt = result["prompt"]
    prompt_id = result["id"]
    response = result["response"]
    print(f"{GREEN}{index}/{total}")
    print(f"{RESET}{prompt}{YELLOW}{response}{RESET}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to an experiment config file. Parameters from this "
             f"file will overwrite those from the default configuration.",
    )

    args = parser.parse_args()
    
    print("Loading config...")
    config_path = args.config
    if config_path is None:
        config_path = DEFAULT_CONFIG
        default_path = None
    else:
        default_path = DEFAULT_CONFIG
        
    config = load_config(config_path, defaults=default_path)
    
    # Load modules
    print("\nLoading modules...")
    from llm_research.training import train_model
    from llm_research.evaluation import run_evaluation
    
    # Train model
    print("\nStarting training...")
    model, tokenizer = train_model(
        config.model.name,
        config.training.data_path,
        config.training.output_path,
        lora_config=config.training.lora,
        sft_config=config.training.sft,
    )
    
    # Evaluate model
    print("\nStarting evaluation...")
    run_evaluation(
        model,
        tokenizer,
        config.evaluation.prompts_path,
        config.evaluation.output_path,
        generation_config=config.evaluation.generation,
        callbacks=(print_data,),
    )
    print("Done.")
    
    
if __name__ == "__main__":
    main()