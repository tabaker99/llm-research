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

from pathlib import Path

from datasets import load_dataset

from openai import OpenAI



SCENARIO_TEMPLATE = ("{context}\n\n"
                     "ACTIONS:\nA. {action1}\nB. {action2}")

ENABLE_TESTING_MODE = True

if not ENABLE_TESTING_MODE:
    client = OpenAI()

def generate_response(prompt, model="gpt-6-astra"):
    if ENABLE_TESTING_MODE:
        return "Dummy response for testing"
    
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text


def generate_prompts(framework_path, template_path, dataset_path):
    moral_framework = Path(framework_path).read_text(encoding="utf-8")
    template = Path(template_path).read_text(encoding="utf-8")

    dataset = load_dataset(
        "csv",
        data_files=dataset_path,
        split="train",
    )

    for row in dataset:
        yield template.format(
            moral_framework=moral_framework,
            scenario=row["context"],
            action1=row["action1"],
            action2=row["action2"],
        )

def generate_training_data(
        dataset_path,
        template_path,
        framework_path,
        gpt_model="gpt-6-astra",
        ):
    moral_framework = Path(framework_path).read_text(encoding="utf-8")
    template = Path(template_path).read_text(encoding="utf-8")

    dataset = load_dataset(
        "csv",
        data_files=dataset_path,
        split="train",
    )
    
    for row in dataset:
        scenario = SCENARIO_TEMPLATE.format(
            context=row["context"],
            action1=row["action1"],
            action2=row["action2"],
        )
        prompt = template.format(
            framework=moral_framework,
            scenario=scenario,
        )
        print(prompt)
        response = generate_response(prompt, model=gpt_model)
        
        yield {"scenario": scenario,
               "response": response}
