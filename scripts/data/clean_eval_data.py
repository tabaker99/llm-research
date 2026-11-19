#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_training_prompts(path):
    prompts = set()

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            text = json.loads(line)["text"]

            if "\nAnswer:" in text:
                question = text.split("\nAnswer:", 1)[0]
                prompts.add((question + "\nAnswer:").strip())

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Remove evaluation prompts that appear in training data."
    )

    parser.add_argument("eval_file")
    parser.add_argument("training_file")
    parser.add_argument(
        "--output",
        default=None,
        help="Output file. Defaults to <eval_file>.filtered.jsonl",
    )

    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    output_path = (
        Path(args.output)
        if args.output
        else eval_path.with_name(eval_path.stem + ".filtered.jsonl")
    )

    training_prompts = load_training_prompts(args.training_file)

    kept = 0
    removed = 0

    with open(eval_path, "r", encoding="utf-8") as source, \
         open(output_path, "w", encoding="utf-8") as destination:

        for line in source:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            prompt = item["prompt"].strip()

            if prompt in training_prompts:
                print(f"Removing duplicate: {prompt!r}")
                removed += 1
                continue

            destination.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\nRemoved: {removed}")
    print(f"Kept:    {kept}")
    print(f"Saved:   {output_path}")


if __name__ == "__main__":
    main()
