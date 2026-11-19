#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:37:28 2026

@author: tbaker
"""


import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import BinaryIO

from datasets import Dataset
from openai import OpenAI
from openai.types import Batch

from .base import CompletionEngine

import logging
logger = logging.getLogger(__name__)

    
class OpenAICompletionEngine(CompletionEngine):
    dummy_response = "Dummy response for testing"
    
    def __init__(self, model="gpt-6-luna"):
        self.model = model
        self.client = OpenAI()
        self.testing_mode = True
        
    def enable_testing_mode(self):
        self.testing_mode = True
        
    def disable_testing_mode(self):
        self.testing_mode = False
        
    def complete(self, prompt):
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
        if self.testing_mode:
            return self.dummy_response
        
        # Get API response
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        output_text = response.output_text
        return output_text


def _write_batch_file(
    batch_file: BinaryIO,
    dataset,
    prompt_field,
    gpt_model,
    endpoint,
    *,
    id_field=None,
) -> None:
    """Write UTF-8 JSONL requests to an open binary file."""
    seen_ids = set()
    for index, row in enumerate(dataset):
        if id_field is None:
            custom_id = f"request-{index}"
        else:
            row_id = row[id_field]
            if not isinstance(row_id, (str, int)) or not str(row_id).strip():
                raise ValueError(
                    f"Invalid ID in field {id_field!r} at row {index}. "
                    "Expected a non-blank string or an integer."
                )
            # Check uniqueness after converting IDs to strings for OpenAI.
            custom_id = str(row_id)
            if custom_id in seen_ids:
                raise ValueError(f"Duplicate batch request ID {custom_id!r}.")
            seen_ids.add(custom_id)

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": endpoint,
            "body": {"model": gpt_model, "input": row[prompt_field]},
        }
        batch_file.write((json.dumps(request) + "\n").encode("utf-8"))


def submit_batch(
    dataset, prompt_field, gpt_model, description=None, *, id_field=None
) -> Batch:
    """Submit one Responses API request per dataset row.

    Parameters
    ----------
    dataset : iterable of mapping
        Rows containing the prompts to submit.
    prompt_field : str
        Field in each row to use as the request input.
    gpt_model : str
        OpenAI model ID to use for every request in the batch.
    description : str, optional
        Description to store in the batch metadata.
    id_field : str, optional
        Field containing unique non-blank string or integer IDs. The value
        is used directly as a string. If omitted, IDs use ``request-<index>``
        with a zero-based row index.

    Returns
    -------
    Batch
        The newly submitted batch object.

    Raises
    ------
    KeyError
        If a row lacks the prompt field or the specified ID field.
    ValueError
        If an ID is invalid or produces a duplicate request ID.
    """
    endpoint = "/v1/responses"
    metadata = (
        {"description": description} if description is not None else None
    )

    with OpenAI() as client:
        with NamedTemporaryFile(suffix=".jsonl") as batch_file:
            _write_batch_file(
                batch_file.file, dataset, prompt_field, gpt_model, endpoint,
                id_field=id_field,
            )
            # Rewind the binary file before passing it to the SDK.
            batch_file.seek(0)
            input_file = client.files.create(
                file=batch_file.file, purpose="batch"
            )

        # Start the batch using the uploaded file's ID.
        return client.batches.create(
            input_file_id=input_file.id,
            endpoint=endpoint,
            completion_window="24h",
            metadata=metadata,
        )


def retrieve_batch(batch_id: str) -> Batch:
    """Fetch the current batch object by ID.

    Parameters
    ----------
    batch_id : str
        ID from the batch object returned by `submit_batch`.

    Returns
    -------
    Batch
        The batch's current status, request counts, and available output
        file ID.
    """
    with OpenAI() as client:
        # Fetch a fresh snapshot so callers can poll for status or file IDs.
        return client.batches.retrieve(batch_id)


def get_batch_status(batch_id: str) -> str:
    """Fetch the current status of a batch by ID."""
    return retrieve_batch(batch_id).status


def download_batch_results(
    batch_id: str,
    output_path: str | Path | None = None,
) -> Dataset:
    """Download the batch output and extract generated text.

    Parameters
    ----------
    batch_id : str
        ID of a batch with an available output file.
    output_path : str or pathlib.Path, optional
        Path where the returned dataset is saved as JSONL. An existing file
        at this path is replaced after a successful write.

    Returns
    -------
    datasets.Dataset
        Batch output rows in file order, including their `custom_id` and a
        `completion` column containing the generated text when available.

    Raises
    ------
    ValueError
        If the batch has no output file.
    """
    rows = []
    with OpenAI() as client:
        batch = client.batches.retrieve(batch_id)
        if batch.output_file_id is None:
            raise ValueError(
                f"Batch {batch_id} has no output file "
                f"(status: {batch.status})."
            )

        # Stream the JSONL because each line is a separate batch result.
        with client.files.with_streaming_response.content(
            batch.output_file_id,
        ) as output:
            for line in output.iter_lines():
                if not line.strip():
                    continue

                # Keep the original result and add text from its messages.
                row = json.loads(line)
                response = row.get("response") or {}
                body = response.get("body") or {}
                output_items = body.get("output")
                row["completion"] = None if output_items is None else "".join(
                    content.get("text") or ""
                    for item in output_items
                    if item["type"] == "message"
                    for content in item["content"]
                    if content["type"] == "output_text"
                )
                rows.append(row)

    dataset = Dataset.from_list(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write beside the destination so a failed export leaves it intact.
        with TemporaryDirectory(dir=output_path.parent) as temp_dir:
            temp_path = Path(temp_dir) / "results.jsonl"
            dataset.to_json(
                temp_path, orient="records", lines=True, force_ascii=False
            )
            temp_path.replace(output_path)

    return dataset
