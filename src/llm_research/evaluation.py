#!/usr/bin/env python3

import json
from pathlib import Path
from contextlib import nullcontext

from llm_research.engine import LLMEngine, load_model


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


def load_prompts(path: str) -> list:
    """
    Load a set of prompts from a file. Each json object in the file should
    contain the keys "id" and "prompt".

    Parameters
    ----------
    path : str
        Path to a .jsonl file containing a set of prompts with prompt ids.

    Returns
    -------
    prompts : list of dict
        List of dict objects. Each dict has keys corresponding to the loaded
        data from the file.

    """
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

            prompts.append(item)

    return prompts



def generate_responses(engine, prompts):
    """
    Generate responses for a collection of prompts.

    Parameters
    ----------
    engine : LLMEngine
        An engine containing an LLM.
    prompts : list
        List containing dicts with an id and a prompt.

    Yields
    ------
    dict
        Contains id, prompt, and response.
    """
    for index, item in enumerate(prompts, start=1):
        prompt_id = item.get("id", str(index))
        prompt = item["prompt"]
        
        # Generate the response
        response = engine.complete(prompt)

        yield {
            "id": prompt_id,
            "prompt": prompt,
            "response": response,
        }


def build_metadata(
        eval_id,
        prompts_path,
        output_path,
    ):
    return {
        "eval_id": eval_id,
        "prompts": str(prompts_path),
        "output": str(output_path),
    }
    

def run_evaluation(
        model,
        tokenizer,
        prompts_path,
        output_path,
        eval_id="",
        generation_config=None,
        callbacks=(),
    ):
    
    prompts = load_prompts(prompts_path)
    engine = LLMEngine(model, tokenizer)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write metadata
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = build_metadata(
        eval_id,
        prompts_path,
        output_path,
    )
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    
    # Generate and save responses
    with open(output_path, "w", encoding="utf-8") as f:
        for index, result in enumerate(
            generate_responses(engine, prompts),
            start=1,
        ):
            record = {
                "id": result["id"],
                "response": result["response"],
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            
            for callback in callbacks:
                callback(result, index, len(prompts))

    print(f"\nSaved outputs to:  {output_path}")
    print(f"Saved metadata to: {metadata_path}")

