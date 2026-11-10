#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:31:01 2026

@author: tbaker
"""

"""Utilities for formatting datasets for supervised fine-tuning."""

from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
from string import Formatter



def load_training_dataset(
    data_path: str | Path,
    template_path: str | Path,
    response_field: str = "response",
) -> Dataset:
    """Load and format a JSONL dataset for prompt-completion training.

    Template placeholders are automatically matched to dataset columns.

    Parameters
    ----------
    data_path : str or Path
        Path to the JSONL dataset.
    template_path : str or Path
        Path to a prompt template using Python format placeholders.
    response_field : str, default="response"
        Dataset column containing the target completion.

    Returns
    -------
    Dataset
        Dataset with ``prompt`` and ``completion`` columns.

    Raises
    ------
    ValueError
        If the dataset is missing required template or response fields.
    """
    dataset = load_dataset(
        "json",
        data_files=str(data_path),
        split="train",
    )

    template = Path(template_path).read_text(encoding="utf-8")

    # Extract required fields from the template
    template_fields = {
        field
        for _, field, _, _ in Formatter().parse(template)
        if field is not None
    }

    required_fields = template_fields | {response_field}
    missing_fields = required_fields - set(dataset.column_names)

    if missing_fields:
        raise ValueError(
            f"Dataset is missing required fields: {sorted(missing_fields)}"
        )

    # Format examples for SFTTrainer
    def format_example(example):
        return {
            "prompt": template.format(**example),
            "completion": example[response_field],
        }

    return dataset.map(
        format_example,
        remove_columns=dataset.column_names,
    )

def load_training_splits(
    train_path: str | Path,
    eval_path: str | Path,
    template_path: str | Path,
    scenario_field: str = "scenario",
    response_field: str = "response",
) -> DatasetDict:
    """Load and format training and evaluation datasets.

    Parameters
    ----------
    train_path : str or Path
        Path to the training JSONL dataset.
    eval_path : str or Path
        Path to the evaluation JSONL dataset.
    template_path : str or Path
        Path to the prompt template.
    scenario_field : str, default="scenario"
        Name of the dataset field containing the scenario.
    response_field : str, default="response"
        Name of the dataset field containing the target response.

    Returns
    -------
    DatasetDict
        Dictionary containing formatted ``train`` and ``eval`` datasets.
    """
    return DatasetDict({
        "train": load_training_dataset(
            train_path, template_path, scenario_field, response_field
        ),
        "eval": load_training_dataset(
            eval_path, template_path, scenario_field, response_field
        ),
    })