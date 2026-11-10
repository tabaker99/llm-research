#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:25:24 2026

@author: tbaker
"""


"""Utilities for filtering paired moral-system training datasets."""

import json
import re
import warnings
from itertools import zip_longest
from pathlib import Path


def extract_response(response):
    """
    Extract reasoning and action from a model response.

    Parameters
    ----------
    response : str
        Response containing REASONING: and ACTION: markers.

    Returns
    -------
    tuple[str, str]
        Cleaned reasoning and action.
    """
    match = re.search(
        r"REASONING:\s*(.*?)\s*ACTION:\s*(.*)",
        response,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError("Missing REASONING or ACTION marker")

    return tuple(
        part.strip(" \t\r\n[]")
        for part in match.groups()
    )


def read_responses(path):
    """
    Yield original JSONL lines and extracted response fields.

    Parameters
    ----------
    path : str or Path
        Input JSONL file.

    Yields
    ------
    tuple[str, object, str, str]
        Original line, scenario, reasoning, and action.
    """
    with open(path, "r", encoding="utf-8", newline="") as file:
        for number, line in enumerate(file, start=1):
            try:
                data = json.loads(line)
                reasoning, action = extract_response(data["response"])
                yield line, data["scenario"], reasoning, action
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"{path}, line {number}: {exc}"
                ) from exc


def path_infix(path, infix):
    """Return the corresponding output path, with an infix added before
    the filename extension."""
    path = Path(path)
    return path.with_name(f"{path.stem}.{infix}.jsonl")


def filter_matching_actions(path_a, path_b):
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
    output_a = path_infix(path_a, "filtered")
    output_b = path_infix(path_b, "filtered")

    if Path(path_a).resolve() == Path(path_b).resolve():
        raise ValueError("Input files must be different")

    if output_a.resolve() == output_b.resolve():
        raise ValueError("Output paths must be different")

    for path in (output_a, output_b):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")

    total = 0
    matched = 0
    invalid = 0

    with (
        open(output_a, "w", encoding="utf-8", newline="") as out_a,
        open(output_b, "w", encoding="utf-8", newline="") as out_b,
    ):
        for total, (item_a, item_b) in enumerate(
            zip_longest(read_responses(path_a), read_responses(path_b)),
            start=1,
        ):
            if item_a is None or item_b is None:
                raise ValueError("Input files have different lengths")

            line_a, scenario_a, _, action_a = item_a
            line_b, scenario_b, _, action_b = item_b

            if scenario_a != scenario_b:
                raise ValueError(f"Scenario mismatch at line {total}")

            if action_a not in {"A", "B"} or action_b not in {"A", "B"}:
                warnings.warn(
                    f"Line {total}: Invalid actions "
                    f"({action_a!r}, {action_b!r})"
                )
                invalid += 1
                continue

            if action_a == action_b:
                out_a.write(line_a)
                out_b.write(line_b)
                matched += 1

    return {
        "total": total,
        "matched": matched,
        "disagreed": total - matched - invalid,
        "invalid": invalid,
        "output_a": output_a,
        "output_b": output_b,
    }