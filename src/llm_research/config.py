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
    """
    Values
    ======
    
    ====================  =====================
    Name                  Type
    ====================  =====================
    ``name``              ``str``
    ``condition``         ``str``
    ====================  =====================
    """
    name: str
    condition: str

@dataclass
class ModelConfig:
    """
    Values
    ======
    
    ========================  =====================
    Name                      Type
    ========================  =====================
    ``name``                  ``str``
    ``enable_quantization``   ``bool``
    ========================  =====================
    """
    name: str
    enable_quantization: bool

@dataclass
class TrainConfig:
    """
    Values
    ======
    
    ====================  =====================
    Name                  Type
    ====================  =====================
    ``data_path``         ``str``
    ``output_path``       ``str``
    ``seed``              ``int``
    ====================  =====================
    
    Structures
    ==========
    
    ====================  =====================
    Name                  Type
    ====================  =====================
    ``lora``              ``LoraConfig``
    ``sft``               ``SFTConfig``
    ====================  =====================
    """

    data_path: str
    output_path: str
    seed: int
    
    lora: LoraConfig
    sft: SFTConfig

@dataclass
class EvalConfig:
    """
    Values
    ======
    
    ====================  =====================
    Name                  Type
    ====================  =====================
    ``prompts_path``      ``str``
    ``output_path``       ``str``
    ``seed``              ``int``
    ====================  =====================
    
    Structures
    ==========
    
    ====================  =====================
    Name                  Type
    ====================  =====================
    ``generation``        ``GenerationConfig``
    ====================  =====================
    """
    prompts_path: str
    output_path: str
    seed: int
    
    generation: GenerationConfig


@dataclass 
class ExperimentConfig:
    """
    Values
    ======
    
    ====================  =====================
    Name                  Type
    ====================  =====================
    ``config_version``    ``int``
    ====================  =====================
    
    Structures
    ==========
    
    ====================  =====================
    Name                  Type
    ====================  =====================
    ``run``               ``RunConfig``
    ``model``             ``ModelConfig``
    ``training``          ``TrainConfig``
    ``evaluation``        ``EvalConfig``
    ====================  =====================
    """
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
        
    # Fix incompatible parameters
    generation_dict = config["evaluation"].get("generation", {})
    if not generation_dict.get("do_sample", False):
        # TODO: Log a warning here
        generation_dict.pop("temperature", None)
        generation_dict.pop("top_p", None)
        generation_dict.pop("top_k", None)
        
        
    experiment_config = ExperimentConfig(
        config_version=config["config_version"],
        run=RunConfig(**config["run"]),
        model=ModelConfig(**config["model"]),
        training=TrainConfig(
            data_path=config["training"]["data_path"],
            output_path =config["training"]["output_path"],
            seed=config["training"]["seed"],
            lora=LoraConfig(**config["training"].get("lora", {})),
            sft=SFTConfig(output_dir=config["training"]["output_path"],
                          **config["training"].get("sft", {})),
        ),
        evaluation=EvalConfig(
            prompts_path=config["evaluation"]["prompts_path"],
            output_path=config["evaluation"]["output_path"],
            seed=config["evaluation"]["seed"],
            generation=GenerationConfig(**generation_dict),
        ),
    )
    return experiment_config