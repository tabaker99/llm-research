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



SCENARIO_TEMPLATE = ("CONTEXT:\n{context}\n\n"
                     "ACTIONS:\nA. {action1}\nB. {action2}")

ENABLE_TESTING_MODE = True

if not ENABLE_TESTING_MODE:
    client = OpenAI()

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


from pathlib import Path

from datasets import load_dataset


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
    scenario : str
        A formatted scenario containing the context and possible actions.
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

    # Construct a scenario for each row
    for row in dataset:
        scenario = scenario_template.format(
            context=row["context"],
            action1=row["action1"],
            action2=row["action2"],
        )

        yield scenario


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
    checkprompt = True
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