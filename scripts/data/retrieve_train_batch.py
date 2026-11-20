#!/usr/bin/env python3
"""Retrieve completed training data from an OpenAI batch."""

import argparse
import json
from pathlib import Path


def main():
    """Check one batch and export its completed dataset when ready."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metadata_file",
        type=Path,
        help="Path to the batch metadata JSON file.",
    )
    args = parser.parse_args()

    from llm_research.data.generation import load_batch_completions
    from llm_research.engine.openai import get_batch_status

    metadata_path = args.metadata_file
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    batch_id = metadata["batch_id"]
    status = get_batch_status(batch_id)
    if status != "completed":
        print(f"Batch {batch_id}: {status}")
        return

    tag = metadata.get("tag")
    if tag is None:
        output_name = f"{metadata_path.stem}_completions.jsonl"
    else:
        output_name = f"{tag}_{batch_id[-8:]}.jsonl"
    output_path = metadata_path.with_name(output_name)

    # Rejoin the downloaded completions to the original training data.
    dataset = load_batch_completions(
        metadata_path=metadata_path, batch_id=batch_id
    )
    # Short ID suffixes can collide, so keep any existing tagged export.
    output = output_path.open("wb" if tag is None else "xb")
    try:
        with output:
            dataset.to_json(output, orient="records", lines=True)
    except BaseException:
        # Remove an incomplete tagged export so retrieval can be retried.
        if tag is not None:
            output_path.unlink(missing_ok=True)
        raise
    print(f"Saved {len(dataset)} rows to {output_path}")


if __name__ == "__main__":
    main()
