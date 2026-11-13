#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:25:53 2026

@author: tbaker
"""

"""
1. Load the dataset.
2. Pick an item from the dataset, and enter it into the template.
3. Pass to ChatGPT.
4. Save response.
"""

import hashlib
import json
import os
import warnings

from pathlib import Path
from datasets import load_dataset
from openai import OpenAI
from tqdm.auto import tqdm

from llm_research.data.formatting import format_row

import logging

logger = logging.getLogger(__name__)


SCENARIO_TEMPLATE = ("CONTEXT:\n{context}\n\n"
                     "ACTIONS:\nA. {action1}\nB. {action2}")

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"

ENABLE_TESTING_MODE = False

if not ENABLE_TESTING_MODE:
    client = OpenAI()
    
def enable_testing_mode():
    global ENABLE_TESTING_MODE
    ENABLE_TESTING_MODE = True

def generate_response(prompt, model="gpt-6-luna"):
    """
    Get a GPT response using the prompt. If global testing mode is enabled,
    a dummy response will be returned without calling the API.

    Parameters
    ----------
    prompt : str
        Prompt to send to GPT.
    model : str, optional
        GPT model to use for generation. The default is "gpt-6-luna".

    Returns
    -------
    output_text : str
        Text response from the API.
    """
    
    # For testing: send a filler response without using the API
    if ENABLE_TESTING_MODE:
        return "Dummy response for testing"
    
    # Get API response
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    output_text = response.output_text
    return output_text



def construct_scenarios(
        dataset_path,
        scenario_template=None,
        ):
    """
    Construct moral scenarios from a dataset.

    Parameters
    ----------
    dataset_path : str
        A .csv file containing moral scenarios and possible actions. The
        dataset must include columns labeled 'context', 'action1', 'action2'.
    scenario_template : str, optional
        A template for formatting the scenario. Must contain the fields:
        `{context}`, `{action1}`, and `{action2}`. If not included, the
        default template `data_generation.SCENARIO_TEMPLATE` will be used.

    Yields
    ------
    scenario_generator : generator
        A generator that yields a formatted scenario containing the context
        and possible actions.
    count : int
        The total number of scenarios.
    """
    
    # Load the dataset
    dataset = load_dataset(
        "csv",
        data_files=dataset_path,
        split="train",
    )

    # Fallback on default scenario template
    if scenario_template is None:
        scenario_template = SCENARIO_TEMPLATE

    def inner_generator():
        # Construct a scenario for each row
        for row in dataset:
            scenario = scenario_template.format(
                context=row["context"],
                action1=row["action1"],
                action2=row["action2"],
            )
    
            yield scenario
    scenario_generator = inner_generator()
    
    return scenario_generator, len(dataset)


def generate_training_data(
        scenarios,
        framework_path,
        template_path,
        gpt_model="gpt-6-luna",
        ):
    """
    Generate training data from moral scenarios using a GPT model.

    Parameters
    ----------
    scenarios : iterable of str
        Moral scenarios to use for generating training responses.
    framework_path : str
        Path to the moral framework file.
    template_path : str
        A prompt template containing the fields `{framework}` and `{scenario}`.
    gpt_model : str, optional
        GPT model used to generate responses.

    Yields
    ------
    dict
        Dictionary containing the scenario and generated response.
    """
    

    # Load template and moral framework
    moral_framework = Path(framework_path).read_text(encoding="utf-8")
    template = Path(template_path).read_text(encoding="utf-8")
    checkprompt = False
    # Generate a response for each scenario
    for scenario in scenarios:
        prompt = template.format(
            framework=moral_framework,
            scenario=scenario,
        )
        
        if checkprompt:
            print(prompt)
            checkprompt=False

        response = generate_response(
            prompt,
            model=gpt_model,
        )

        yield {
            "scenario": scenario,
            "response": response,
        }
        
def generate_completions_legacy(
        dataset,
        engine,
        log_function=None,
        completion_field="completion",
        *,
        template_path=None,
        prompt_field=None,
        template_vars=None,
        ):
    """
    Generate completions for prompts from a dataset.

    Prompts are either taken from an existing dataset column or constructed
    using a template. Formatted prompts are not stored in the dataset.

    Parameters
    ----------
    dataset : datasets.Dataset
        Dataset containing the input data.
    engine : CompletionEngine
        Engine providing a `complete(prompt)` method.
    log_function : function, optional
        Function that accepts a row, prompt, and completion and returns a
        formatted string for logging.
    completion_field : str, default="completion"
        Column where completions will be saved.
    template_path : str or Path, optional
        Path to a prompt template. Mutually exclusive with `prompt_field`.
    prompt_field : str, optional
        Column containing prompts. Mutually exclusive with `template_path`.
    template_vars : dict, optional
        Additional variables available when formatting the template.
        Dataset columns take precedence over these variables.

    Returns
    -------
    datasets.Dataset
        Dataset with the generated completions added.
    """

    if (prompt_field is None) == (template_path is None):
        raise ValueError(
            "Specify exactly one of prompt_field or template_path."
        )

    template_vars = template_vars or {}

    # Load template once.
    template = (
        Path(template_path).read_text(encoding="utf-8")
        if template_path is not None
        else None
    )

    def process(row):
        # Construct prompt without storing it in the dataset.
        if template is not None:
            prompt = format_row(
                row,
                template,
                "prompt",
                template_vars,
            )["prompt"]
        else:
            prompt = row[prompt_field]


        # Generate completion.
        completion = engine.complete(prompt)

        # Log selected fields.
        if log_function is not None:
            tqdm.write(log_function(row, prompt, completion))
        else:
            tqdm.write(f"Completion:\n{completion}")

        return {completion_field: completion}

    return dataset.map(
        process,
        desc="Generating completions",
    )





def _hash_value(value):
    """Compute a SHA-256 hash of a JSON-serializable value."""
    data = json.dumps(
        value, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _validate_partial_results(
    dataset,
    partial_path,
    saved_metadata,
    metadata,
    completion_field,
):
    """
    Validate saved metadata and previously completed rows.

    The final saved line is excluded from validation because it will
    be regenerated. This function does not modify any files.

    Returns the number of validated rows.
    """
    # Strict configuration checks.
    strict_fields = (
        "completion_field",
        "prompt_field",
        "template_hash",
        "template_vars_hash",
    )

    for field in strict_fields:
        if saved_metadata.get(field) != metadata[field]:
            raise ValueError(
                f"Partial results metadata mismatch: {field}"
            )

    # Warn about possible dataset changes.
    for field in ("dataset_fingerprint", "dataset_length"):
        if saved_metadata.get(field) != metadata[field]:
            warnings.warn(
                f"{field} differs from the previous run.",
                stacklevel=2,
            )

    # Validate all saved rows except the last.
    with partial_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines[:-1]):
        saved_row = json.loads(line)

        if i >= len(dataset):
            raise ValueError(
                "Partial results contain more completed rows "
                "than the current dataset."
            )

        if completion_field not in saved_row:
            raise ValueError(
                f"Missing completion field in partial row {i}."
            )

        saved_row.pop(completion_field)

        if saved_row != dataset[i]:
            raise ValueError(
                f"Dataset mismatch at partial row {i}."
            )

    return max(0, len(lines) - 1)


def _prepare_partial_results(
    dataset,
    partial_path,
    metadata_path,
    metadata,
    completion_field,
    resume=True,
):
    """
    Prepare partial results for generation or resumption.

    Validates previously completed rows before discarding the final
    saved line. Returns the index of the next row to generate.
    """
    if not resume:
        partial_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    if partial_path.exists() != metadata_path.exists():
        raise ValueError(
            "Incomplete partial results: expected both the "
            "partial data file and its metadata file."
        )

    # Start a new run.
    if not partial_path.exists():
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        partial_path.touch()
        return 0

    # Load saved metadata.
    with metadata_path.open("r", encoding="utf-8") as f:
        saved_metadata = json.load(f)

    # Validate without modifying the partial file.
    completed = _validate_partial_results(
        dataset,
        partial_path,
        saved_metadata,
        metadata,
        completion_field,
    )

    # Validation succeeded: discard the final saved line.
    with partial_path.open("rb+") as f:
        last_position = None

        while True:
            position = f.tell()
            if not f.readline():
                break
            last_position = position

        if last_position is not None:
            f.truncate(last_position)

    tqdm.write(f"Resuming from row {completed}.")

    return completed


def generate_completions(
    dataset,
    engine,
    output_path,
    log_function=None,
    completion_field="completion",
    *,
    template_path=None,
    prompt_field=None,
    template_vars=None,
    resume=True,
):
    """
    Generate completions for a dataset and save them incrementally.

    Prompts are either obtained from an existing dataset column or
    constructed using a template. Formatted prompts are not stored
    in the dataset.

    Completions are written to a partial JSONL file after each
    generation. Interrupted runs can resume from these results after
    validating the saved data and configuration. The last saved line
    is discarded and regenerated only after validation succeeds.

    Upon completion, the partial file is renamed to the requested
    output path, and the associated metadata file is deleted.

    Parameters
    ----------
    dataset : datasets.Dataset
        Input dataset containing prompts or fields referenced by the
        template. All original columns are preserved in the output.
    engine : CompletionEngine
        Completion engine implementing ``complete(prompt)`` and
        returning a string.
    output_path : str or pathlib.Path
        Destination JSONL file. Parent directories are created if
        necessary. An existing output file is replaced only after
        generation finishes successfully.
    log_function : callable, optional
        Function accepting ``(row, prompt, completion)`` and
        returning a formatted string for console logging. If omitted,
        the completion is printed directly.
    completion_field : str, default="completion"
        Name of the column containing generated completions.
        Must not already exist in the input dataset.
    template_path : str or pathlib.Path, optional
        Path to a UTF-8 prompt template. Replacement fields are
        populated using dataset columns and ``template_vars``.
        Mutually exclusive with ``prompt_field``.
    prompt_field : str, optional
        Name of an existing dataset column containing prompts.
        Mutually exclusive with ``template_path``.
    template_vars : dict, optional
        Additional values available when formatting the template.
        Dataset columns take precedence over these values.
        Values must be JSON-serializable for resume validation.
    resume : bool, default=True
        If True, resume from existing partial results when available.
        Saved configuration and completed rows are validated first.
        If False, discard any existing partial results and start over.

    Returns
    -------
    datasets.Dataset
        Dataset containing the original columns and generated
        completions, loaded from the finalized JSONL file.

    Raises
    ------
    ValueError
        If prompt arguments are invalid, the completion field already
        exists, or the saved results cannot be safely resumed.
    KeyError
        If a requested prompt field or template variable is missing.
    FileNotFoundError
        If the template file cannot be found.
    TypeError
        If template variables cannot be JSON-serialized.

    Notes
    -----
    The partial results and metadata are stored alongside the output
    as ``<stem>.partial.jsonl`` and ``<stem>.resume.json``.

    Resume validation requires matching prompt configuration and
    unchanged dataset rows for previously completed examples.
    Changes to the dataset fingerprint or length produce warnings,
    provided previously completed rows still match.

    The final saved line is always regenerated when resuming, even
    when it is valid. Validation failures leave existing partial
    results unchanged.

    The caller is responsible for maintaining consistent model
    weights and generation settings across resumed runs.

    Results are flushed after every completion, protecting against
    ordinary process interruptions but not guaranteeing durability
    against power failure.
    """
    if (prompt_field is None) == (template_path is None):
        raise ValueError(
            "Specify exactly one of prompt_field or template_path."
        )

    if completion_field in dataset.column_names:
        raise ValueError(
            f"Completion field {completion_field!r} already exists."
        )

    if prompt_field is not None and prompt_field not in dataset.column_names:
        raise KeyError(f"Prompt field not found: {prompt_field}")

    template_vars = template_vars or {}

    # Load template once.
    template = (
        Path(template_path).read_text(encoding="utf-8")
        if template_path is not None
        else None
    )

    # Set up output paths.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    partial_path = output_path.with_name(
        output_path.stem + ".partial.jsonl"
    )
    metadata_path = output_path.with_name(
        output_path.stem + ".resume.json"
    )

    # Metadata for resume validation.
    metadata = {
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_length": len(dataset),
        "completion_field": completion_field,
        "prompt_field": prompt_field,
        "template_hash": _hash_value(template),
        "template_vars_hash": _hash_value(template_vars),
    }

    # Initialize or resume partial results.
    completed = _prepare_partial_results(
        dataset,
        partial_path,
        metadata_path,
        metadata,
        completion_field,
        resume=resume,
    )

    # Generate remaining completions.
    with partial_path.open("a", encoding="utf-8") as f:
        for row in tqdm(
            dataset.select(range(completed, len(dataset))),
            initial=completed,
            total=len(dataset),
            desc="Generating completions",
        ):
            # Construct prompt.
            if template is not None:
                prompt = format_row(
                    row, template, "prompt", template_vars
                )["prompt"]
            else:
                prompt = row[prompt_field]

            # Generate completion.
            completion = engine.complete(prompt)

            # Save original fields and completion.
            result = {**row, completion_field: completion}
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            # Print logging information.
            if log_function is not None:
                tqdm.write(log_function(row, prompt, completion))
            else:
                tqdm.write(f"Completion:\n{completion}")

    # Finalize results.
    os.replace(partial_path, output_path)
    metadata_path.unlink()

    return load_dataset(
        "json",
        data_files=str(output_path),
        split="train",
    )