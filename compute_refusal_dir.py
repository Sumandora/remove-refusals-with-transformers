import jaxtyping

import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig

import einops

from tqdm import tqdm

torch.inference_mode()

torch.set_default_device("cpu")

#MODEL_ID = "stabilityai/stablelm-2-1_6b"
#MODEL_ID = "stabilityai/stablelm-2-zephyr-1_6b"
#MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
#MODEL_ID = "Qwen/Qwen-1_8B-chat"
MODEL_ID = "google/gemma-1.1-2b-it"
#MODEL_ID = "google/gemma-1.1-7b-it"
#MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16))
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

instructions = 32
layer_idx = int(len(model.model.layers) * 0.6)
pos = -1

print("Instruction count: " + str(instructions))
print("Layer index: " + str(layer_idx))

with open("harmful.txt", "r") as f:
    harmful = f.readlines()

with open("harmless.txt", "r") as f:
    harmless = f.readlines()

harmful_instructions = random.sample(harmful, instructions)
harmless_instructions = random.sample(harmless, instructions)

harmful_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}], add_generation_prompt=True,
                                  return_tensors="pt") for insn in harmful_instructions]
harmless_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}], add_generation_prompt=True,
                                  return_tensors="pt") for insn in harmless_instructions]

max_its = instructions*2
bar = tqdm(total=max_its)

def generate(toks):
    bar.update(n=1)
    return model.generate(toks.to(model.device), use_cache=False, max_new_tokens=1, return_dict_in_generate=True, output_hidden_states=True)

harmful_outputs = [generate(toks) for toks in harmful_toks]
harmless_outputs = [generate(toks) for toks in harmless_toks]

bar.close()

harmful_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmful_outputs]
harmless_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmless_outputs]

print(harmful_hidden)

harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
harmless_mean = torch.stack(harmless_hidden).mean(dim=0)

print(harmful_mean)

refusal_dir = harmful_mean - harmless_mean
refusal_dir = refusal_dir / refusal_dir.norm()

print(refusal_dir)

torch.save(refusal_dir, MODEL_ID.replace("/", "_") + "_refusal_dir.pt")
