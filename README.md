# llm-research

## Purpose ##
To determine the effects of an internally consistent/inconsistent moral system on the alignment of an LLM.

## Experiment outline ##
1. Design a consistent and an inconsistent moral system
2. Train separate models on each system using the Constitutional AI (CAI) method
3. Compare the moral behaviors of the two models

## Current progress ##
* Implemented supervised fine-tuning on Qwen3-0.6b on my local hardware.
* Prepared a training/evaluation pipeline that follows a configuration file.
* Designed a consistent and an inconsistent moral system.
* Next steps: generate training data for the CAI method.
