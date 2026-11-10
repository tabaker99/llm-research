#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 19:23:51 2026

@author: tbaker
"""

import argparse



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
    
    
    
    from llm_research.config import load_config
    
    
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
    from llm_research.data.formatting import load_training_dataset
    
    
    # Load dataset
    dataset = load_training_dataset(
        data_path=config.training.dataset.path,
        dataset_format=config.training.dataset.format,
        template_path=config.training.dataset.template_path,
        text_field=config.training.dataset.text_field,
        prompt_field=config.training.dataset.prompt_field,
        completion_field=config.training.dataset.completion_field,
    )
    
    # Train model
    print("\nStarting training...")
    model, tokenizer = train_model(
        config.model.name,
        dataset,
        config.training.output_path,
        enable_quantization=config.model.enable_quantization,
        lora_config=config.training.lora,
        seed=config.training.seed,
        sft_config=config.training.sft,
    )
    
    # Evaluate model
    if config.evaluation.skip_eval:
        print("\nSkipping evaluation.")
    else:
        print("\nStarting evaluation...")
        run_evaluation(
            model,
            tokenizer,
            config.evaluation.prompts_path,
            config.evaluation.output_path,
            generation_config=config.evaluation.generation,
            seed=config.evaluation.seed,
            callbacks=(print_data,),
        )
    print("Done.")
    
    
if __name__ == "__main__":
    main()