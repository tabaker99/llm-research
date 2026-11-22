#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:25:24 2026

@author: tbaker
"""


"""Utilities for filtering paired moral-system training datasets."""

import json
import warnings
from itertools import zip_longest
from pathlib import Path

from llm_research.data.processing import split_string


import logging
logger = logging.getLogger(__name__)


def read_responses(path, response_field="response", id_field="id"):
    """
    Yield original JSONL lines, the scenario, and the reasoning/action
    extracted from the response.

    Parameters
    ----------
    path : str or Path
        Input JSONL file.
    response_field : str
        Field that contains the response. The reasoning and action will be
        extracted from this field.
    id_field : str
        Field that contains a unique scenario id.

    Yields
    ------
    tuple[str, object, str, str]
        Original line, id, reasoning, and action.
    """
    with open(path, "r", encoding="utf-8", newline="") as file:
        for number, line in enumerate(file, start=1):
            try:
                data = json.loads(line)
                reasoning, action = split_string(data[response_field]).values()
                yield line, data[id_field], reasoning, action
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"{path}, line {number}: {exc}"
                ) from exc


def _path_infix(path, infix):
    """Return the corresponding output path, with an infix added before
    the filename extension."""
    path = Path(path)
    return path.with_name(f"{path.stem}.{infix}.jsonl")


def filter_matching_actions(
        path_a,
        path_b,
        overwrite=False,
        response_field="response",
        id_field="id",
        condition_function=None,
    ):
    """
    Retain paired examples whose actions agree and are valid.

    The two datasets must contain identical scenarios in the
    same order. Original JSONL lines are preserved unchanged.

    Parameters
    ----------
    path_a, path_b : str or Path
        Input JSONL files.

    Returns
    -------
    dict
        Counts and output file paths.
    """
    output_a = _path_infix(path_a, "filtered")
    output_b = _path_infix(path_b, "filtered")

    condition_function = condition_function or (lambda a, b: True)

    if Path(path_a).resolve() == Path(path_b).resolve():
        raise ValueError("Input files must be different")

    if output_a.resolve() == output_b.resolve():
        raise ValueError("Output paths must be different")

    if not overwrite:
        for path in (output_a, output_b):
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}")

    total = 0
    matched = 0
    broken_condition = 0
    invalid = 0

    responses_a = read_responses(path_a,
                                 response_field=response_field,
                                 id_field=id_field)
    responses_b = read_responses(path_b,
                                 response_field=response_field,
                                 id_field=id_field)

    with (
        open(output_a, "w", encoding="utf-8", newline="") as out_a,
        open(output_b, "w", encoding="utf-8", newline="") as out_b,
    ):
        for total, (item_a, item_b) in enumerate(
            zip_longest(responses_a, responses_b),
            start=1,
        ):
            if item_a is None or item_b is None:
                raise ValueError("Input files have different lengths")

            line_a, id_a, reasoning_a, action_a = item_a
            line_b, id_b, reasoning_b, action_b = item_b

            if id_a != id_b:
                raise ValueError(f"ID mismatch at line {total}")

            if action_a not in {"A", "B"} or action_b not in {"A", "B"}:
                logger.warning(
                    f"Line {total}: Invalid actions "
                    f"({action_a!r}, {action_b!r})"
                )
                invalid += 1
                continue

            if (action_a == action_b):
                if condition_function(reasoning_a, reasoning_b):
                    out_a.write(line_a)
                    out_b.write(line_b)
                    matched += 1
                else:
                    logger.info(f"Broken condition at line {total}.")
                    broken_condition += 1
            else:
                logger.info(f"Different actions at line {total}.")

    return {
        "total": total,
        "matched": matched,
        "disagreed": total - matched - invalid - broken_condition,
        "broken_condition": broken_condition,
        "invalid": invalid,
        "output_a": output_a,
        "output_b": output_b,
    }
