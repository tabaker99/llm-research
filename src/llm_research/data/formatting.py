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


import logging
logger = logging.getLogger(__name__)


def apply_template(
        dataset,
        template,
        output_column,
        remove_source_columns=False,
        ):
    """
    Create a new dataset column by formatting existing columns with a template.

    Template fields are interpreted as dataset column names. For each row,
    the values from those columns are inserted into the template and the
    resulting string is stored in `output_column`.

    Parameters
    ----------
    dataset : datasets.Dataset
        Hugging Face dataset containing the columns referenced by `template`.
    template : str
        Format string whose replacement fields correspond to dataset column
        names. For example, ``"{context}\\nAction A: {action1}"``.
    output_column : str
        Name of the new column containing the formatted strings.
    remove_source_columns : bool, optional
        If True, remove the columns referenced by the template after creating
        the new column. If False, retain them. The default is False.

    Returns
    -------
    datasets.Dataset
        A new dataset containing `output_column` and, unless removed, the
        original source columns.

    Raises
    ------
    KeyError
        If a field referenced by the template does not exist in the dataset.
    """

    # Extract the replacement-field names from the format string.
    # For example, "{context} {action1}" -> ["context", "action1"].
    fields = [
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    ]

    # Remove duplicates while preserving their original order.
    fields = list(dict.fromkeys(fields))

    # Validate that every template field corresponds to a dataset column.
    missing = [field for field in fields if field not in dataset.column_names]
    if missing:
        raise KeyError(f"Template fields not found in dataset: {missing}")

    def format_row(row):
        # Select only the values required by the template.
        values = {field: row[field] for field in fields}

        # Dataset.map() merges this returned dictionary into the row.
        return {
            output_column: template.format(**values)
        }

    # Optionally discard the source columns after constructing the new one.
    remove_columns = fields if remove_source_columns else None

    return dataset.map(
        format_row,
        remove_columns=remove_columns,
    )


def load_training_dataset(
    data_path: str | Path,
    dataset_format: str,
    template_path: str | Path | None = None,
    text_field: str | None = None,
    prompt_field: str | None = None,
    completion_field: str | None = None,
) -> Dataset:
    """
    Load and format a dataset for supervised fine-tuning.

    Parameters
    ----------
    data_path : str or Path
        Path to the JSONL dataset.
    dataset_format : str
        Format of the training dataset. Must be ``"text"`` or
        ``"prompt_completion"``.
    template_path : str or Path or None, default=None
        Optional template with placeholders matching dataset columns. For
        text datasets, the formatted result is stored in ``"text"``. For
        prompt-completion datasets, it is stored in ``"prompt"``.
    text_field : str or None, default=None
        Existing column to rename to ``"text"`` when no template is used.
    prompt_field : str or None, default=None
        Existing column to rename to ``"prompt"`` when no template is used.
    completion_field : str or None, default=None
        Existing column to rename to ``"completion"``. This rename is
        performed whether or not a template is used.

    Returns
    -------
    Dataset
        Loaded and formatted training dataset.

    Raises
    ------
    ValueError
        If `dataset_format` is not recognized.
    """

    dataset = load_dataset(
        "json",
        data_files=str(data_path),
        split="train",
    )

    if dataset_format not in ("text", "prompt_completion"):
        raise ValueError(f"Unknown dataset format: {dataset_format}")

    # Apply template to the appropriate training field.
    if template_path:
        template = Path(template_path).read_text(encoding="utf-8")

        output_field = (
            "text"
            if dataset_format == "text"
            else "prompt"
        )

        dataset = apply_template(
            dataset,
            template,
            output_field,
        )

    # Without a template, rename the specified source field.
    elif dataset_format == "text" and text_field:
        if text_field != "text":
            dataset = dataset.rename_column(text_field, "text")

    elif dataset_format == "prompt_completion" and prompt_field:
        if prompt_field != "prompt":
            dataset = dataset.rename_column(prompt_field, "prompt")

    # Completion is independent of prompt formatting, so always rename it.
    if completion_field and completion_field != "completion":
        dataset = dataset.rename_column(
            completion_field,
            "completion",
        )

    return dataset

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