#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 19:23:51 2026

@author: tbaker
"""

import argparse

from llm_research.config import load_config


DEFAULT_CONFIG = "./configs/DEFAULT.toml"



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
    train_model(
        config.model.name,
        config.training.data_path,
        config.training.output_path,
    )
    
    # TODO: Either delete the model to save space, or reuse the model.
    
    # Evaluate model
    print("\nStarting evaluation...")
    run_evaluation(
        config.model.name,
        config.training.output_path,
        config.evaluation.prompts_path,
        config.evaluation.output_path,
        generation_config=config.evaluation.generation,
    )
    print("Done.")
    
    
if __name__ == "__main__":
    main()