#!/usr/bin/env python3

import argparse
import json
from pathlib import Path



DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
DEFAULT_ADAPTER = None
DEFAULT_PROMPTS = "data/toy_eval_prompts.jsonl"
DEFAULT_OUTPUT = "outputs/results/toy_eval.jsonl"

# Colors for console output!
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"


def load_prompts(path):
    prompts = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            if "prompt" not in item:
                raise ValueError(
                    f"Missing 'prompt' field on line {line_number}"
                )

            prompts.append(item["prompt"])

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Run a set of prompts through an LLM and save the outputs."
    )

    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        help="Path to a LoRA adapter. Omit to use the base model.",
    )

    parser.add_argument(
        "--prompts",
        default=DEFAULT_PROMPTS,
        help=f"Prompt JSONL file (default: {DEFAULT_PROMPTS})",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Base model name/path (default: {DEFAULT_MODEL})",
    )

    args = parser.parse_args()


    from llm_research.llm_engine import LLMEngine

    prompts = load_prompts(args.prompts)

    output_path = Path(args.output)
    metadata_path = output_path.with_suffix(".metadata.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": args.model,
        "adapter": args.adapter,
        "prompts": args.prompts,
        "output": str(output_path),
    }

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"Loading model: {args.model}")

    if args.adapter:
        print(f"Loading adapter: {args.adapter}")
    else:
        print("No adapter: evaluating base model")

    engine = LLMEngine(
        args.model,
        adapter_path=args.adapter,
    )

    print(f"Running {len(prompts)} prompts...\n")

    with open(output_path, "w", encoding="utf-8") as output_file:
        for index, prompt in enumerate(prompts, start=1):
            print(f"{GREEN}[{index}/{len(prompts)}]{RESET}")
            print(prompt, end="", flush=True)

            response = engine.complete(prompt)

            print(f"{YELLOW}{response}{RESET}")

            # Each line contains only the model's output as a JSON string.
            output_file.write(
                json.dumps(response, ensure_ascii=False) + "\n"
            )
            output_file.flush()

    print(f"\nSaved outputs to:  {output_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
