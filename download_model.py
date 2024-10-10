import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

base_save_dir = "/data/models"
model_name_for_path = model_name.replace("/", "--")
save_path = os.path.join(base_save_dir, model_name_for_path)

os.makedirs(save_path, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model saved to: {save_path}")