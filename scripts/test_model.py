
#!/usr/bin/env python3


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    )

prompt = "The capital of France is"
#prompt = "Hello World"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
    )

generated = output[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))


