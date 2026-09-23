#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 19:59:36 2026

@author: tbaker
"""

import tomllib
from dataclasses import dataclass
from peft import LoraConfig
from trl import SFTConfig
from transformers import GenerationConfig

    
@dataclass
class RunConfig:
    name: str
    condition: str

@dataclass
class ModelConfig:
    name: str

@dataclass
class TrainConfig:
    data_path: str
    output_path: str
    seed: int
    
    lora: LoraConfig
    sft: SFTConfig

@dataclass
class EvalConfig:
    prompts_path: str
    output_path: str
    seed: int
    
    generation: GenerationConfig


@dataclass 
class ExperimentConfig:
    config_version: int
    
    run: RunConfig
    model: ModelConfig
    training: TrainConfig
    evaluation: EvalConfig


def _deepmerge_dicts(base, override):
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deepmerge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def load_config(cfgpath, defaults=None):
    with open(cfgpath, "rb") as f:
        config = tomllib.load(f)
    
    # If defaults are provided, merge!
    if defaults is not None:
        with open(defaults, "rb") as f:
            defaults_config = tomllib.load(f)
        
        config = _deepmerge_dicts(defaults_config, config)
        
    experiment_config = ExperimentConfig(
        config_version=config["config_version"],
        run=RunConfig(**config["run"]),
        model=ModelConfig(**config["model"]),
        training=TrainConfig(
            data_path=config["training"]["data_path"],
            output_path =config["training"]["output_path"],
            seed=config["training"]["seed"],
            lora=LoraConfig(**config["training"].get("lora", {})),
            sft=SFTConfig(**config["training"].get("sft", {})),
        ),
        evaluation=EvalConfig(
            prompts_path=config["evaluation"]["prompts_path"],
            output_path=config["evaluation"]["output_path"],
            seed=config["evaluation"]["seed"],
            generation=GenerationConfig(
                **config["evaluation"].get("generation", {})
            )
        ),
    )
    return experiment_config