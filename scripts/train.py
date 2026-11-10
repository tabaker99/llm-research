#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 22:30:24 2026

@author: tbaker
"""

from llm_research.training import train_model


MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATA_PATH = "data/toy_train.jsonl"
OUTPUT_DIR = "outputs/adapters/toy"


if __name__ == "__main__":
    train_model(MODEL_NAME, DATA_PATH, OUTPUT_DIR)
    