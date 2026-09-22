#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:34:33 2026

@author: tbaker
"""

import torch

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer



def train_model(model_name, data_path, output_dir):
    dataset = load_dataset(
        "json",
        data_files=data_path,
        split="train",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    training_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=128,

        num_train_epochs=10,
        per_device_train_batch_size=1,
        learning_rate=2e-4,

        fp16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
