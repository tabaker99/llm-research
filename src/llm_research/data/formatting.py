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


def format_row(row, template, field, template_vars=None):
    values = {**(template_vars or {}), **row}
    return {field: template.format(**values)}

def apply_template(
        dataset,
        output_column,
        template=None,
        template_path=None,
        template_vars=None,
        remove_source_columns=False,
        ):
    """
    Create a dataset column by formatting rows with a template.

    Template fields may reference dataset columns or variables supplied
    through `template_vars`. Dataset columns take precedence.

    Parameters
    ----------
    dataset : datasets.Dataset
        Dataset containing values referenced by the template.
    output_column : str
        Name of the column containing the formatted strings.
    template : str, optional
        Template format string.
    template_path : str or Path, optional
        Path to a template file. Mutually exclusive with `template`.
    template_vars : dict, optional
        Additional values available to the template.
    remove_source_columns : bool, default=False
        Whether to remove dataset columns referenced by the template.

    Returns
    -------
    datasets.Dataset
        Dataset with the formatted output column.

    Raises
    ------
    TypeError
        If neither or both template sources are provided.
    KeyError
        If a template field is unavailable.
    """

    if (template is None) == (template_path is None):
        raise TypeError(
            "Specify exactly one of template or template_path."
        )

    if template_path is not None:
        template = Path(template_path).read_text(encoding="utf-8")

    template_vars = template_vars or {}

    # Extract template fields.
    fields = list(dict.fromkeys(
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    ))

    # Validate fields against dataset columns and external variables.
    available = set(dataset.column_names) | set(template_vars)
    missing = [field for field in fields if field not in available]

    if missing:
        raise KeyError(f"Template fields not found: {missing}")

    # Only remove fields that are actual dataset columns.
    remove_columns = (
        [field for field in fields if field in dataset.column_names]
        if remove_source_columns
        else None
    )

    return dataset.map(
        format_row,
        fn_kwargs={
            "template": template,
            "field": output_column,
            "template_vars": template_vars,
        },
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