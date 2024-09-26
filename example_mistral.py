# Mistral model with KIVI
import warnings
warnings.filterwarnings("ignore")
import torch
import random
from models.mistral_kivi import MistralForCausalLM_KIVI
from transformers import MistralConfig, AutoTokenizer
from datasets import load_dataset

# For reproducibility
random.seed(0)
torch.manual_seed(0)

config = MistralConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

config.k_bits = 2 # KiVi currently support 2/4 K/V bits
config.v_bits = 2
config.group_size = 32 
config.residual_length = 32 # corresponding to the number of recent fp16 tokens
CACHE_DIR = "./"

model = MistralForCausalLM_KIVI.from_pretrained(
    pretrained_model_name_or_path='mistralai/Mistral-7B-Instruct-v0.3',
    config=config,
    cache_dir=CACHE_DIR,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()

enc = AutoTokenizer.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.3', 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='mistral')

dataset = load_dataset('gsm8k', 'main')

prompt = ''
for i in range(5):
    prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
inputs = enc(prompt, return_tensors="pt").input_ids.cuda()

output = model.generate(inputs, max_new_tokens=96)
config_str = f"# prompt tokens: {inputs.shape[1]}, K bit: {config.k_bits}, v_bits: {config.v_bits}, group_size: {config.group_size}, residual_length: {config.residual_length}"

print(prompt + "\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nKiVi Output:")
print(enc.decode(output[0].tolist()[inputs.shape[1]:], skip_special_tokens=True))