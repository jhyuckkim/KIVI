import torch
from transformers import AutoTokenizer, AutoConfig
import lm_eval
import os
import json

from models.mistral_kivi import MistralForCausalLM_KIVI

# Define model arguments (adjust these based on your requirements)
k_bits = 4  # Example value (could be 2, 4, or 16)
v_bits = 4  # Example value (could be 2, 4, or 16)
group_size = 64  # Define if needed
residual_length = 128  # Define if needed
dtype = torch.float16
low_cpu_mem_usage = True

MODEL_PATH = '/data/models/mistralai--Mistral-7B-Instruct-v0.3'
SAVE_PATH = '/lm_eval_results'
RESULTS_FILE_NAME = "Mistral-7B-Instruct-v0.3.json"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False,
    trust_remote_code=True,
)

config = AutoConfig.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)
config.k_bits = k_bits
config.v_bits = v_bits
config.group_size = group_size
config.residual_length = residual_length
config.attention_dropout = 0.0

model = MistralForCausalLM_KIVI.from_pretrained(
    MODEL_PATH,
    config=config,
    cache_dir=MODEL_PATH,
    torch_dtype=dtype,
    low_cpu_mem_usage=low_cpu_mem_usage,
)

model.eval().to('cuda')
# model.tie_weights()

# Corrected variable name from 'custom_model' to 'model'
lm_obj = lm_eval.models.huggingface.HFLM(
    pretrained=model,  # Pass your custom model instance
    tokenizer=tokenizer,
    device="cuda",
)

task_manager = lm_eval.tasks.TaskManager()
tasks = ["gsm8k"]

task_dict = lm_eval.tasks.get_task_dict(
    tasks,
    task_manager
)

print("task_dict", task_dict)

results = lm_eval.evaluate(
    lm=lm_obj,
    task_dict=task_dict,
)

print(results["results"])

# Save results if the path is specified
if SAVE_PATH:
    os.makedirs(SAVE_PATH, exist_ok=True)
    results_file = os.path.join(SAVE_PATH, results_file_name)
    with open(results_file, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {results_file}")