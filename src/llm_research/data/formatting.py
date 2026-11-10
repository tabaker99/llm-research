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


from pathlib import Path

from datasets import Dataset, load_dataset

def load_text_dataset(
    data_path: str | Path,
    template_path: str | Path | None = None,
    text_field: str = "text",
) -> Dataset:
    """Load and format a JSONL dataset for text-only training.

    Parameters
    ----------
    data_path : str or Path
        Path to the JSONL dataset.
    template_path : str or Path or None, default=None
        Optional text template with placeholders matching dataset columns.
    text_field : str, default="text"
        Column containing the training text when no template is provided.

    Returns
    -------
    Dataset
        Dataset containing a ``text`` column.
    """
    dataset = load_dataset(
        "json",
        data_files=str(data_path),
        split="train",
    )

    if template_path:
        template = Path(template_path).read_text(encoding="utf-8")

        def format_example(example):
            return {"text": template.format(**example)}

        return dataset.map(
            format_example,
            remove_columns=dataset.column_names,
        )

    return dataset.select_columns([text_field]).rename_column(
        text_field, "text"
    ) if text_field != "text" else dataset.select_columns(["text"])

def load_prompt_completion_dataset(
    data_path: str | Path,
    template_path: str | Path | None = None,
    prompt_field: str = "prompt",
    completion_field: str = "completion",
) -> Dataset:
    """Load and format a JSONL dataset for prompt-completion training.

    Parameters
    ----------
    data_path : str or Path
        Path to the JSONL dataset.
    template_path : str or Path or None, default=None
        Optional prompt template with placeholders matching dataset columns.
    prompt_field : str, default="prompt"
        Column containing the prompt when no template is provided.
    completion_field : str, default="completion"
        Column containing the target completion.

    Returns
    -------
    Dataset
        Dataset with ``prompt`` and ``completion`` columns.
    """
    dataset = load_dataset(
        "json",
        data_files=str(data_path),
        split="train",
    )

    template = (
        Path(template_path).read_text(encoding="utf-8")
        if template_path else None
    )

    def format_example(example):
        return {
            "prompt": (
                template.format(**example)
                if template is not None
                else example[prompt_field]
            ),
            "completion": example[completion_field],
        }

    return dataset.map(
        format_example,
        remove_columns=dataset.column_names,
    )


def load_training_dataset(
    data_path: str | Path,
    dataset_format: str,
    template_path: str | Path | None = None,
    text_field: str = "text",
    prompt_field: str = "prompt",
    completion_field: str = "completion",
) -> Dataset:
    """Load and format a dataset for supervised fine-tuning."""
    if dataset_format == "text":
        return load_text_dataset(
            data_path=data_path,
            template_path=template_path,
            text_field=text_field,
        )

    if dataset_format == "prompt_completion":
        return load_prompt_completion_dataset(
            data_path=data_path,
            template_path=template_path,
            prompt_field=prompt_field,
            completion_field=completion_field,
        )

    raise ValueError(f"Unknown dataset format: {dataset_format}")

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