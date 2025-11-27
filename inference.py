import einops
import jaxtyping
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from inspect import signature

torch.inference_mode()

MODEL_ID = "tiiuae/Falcon3-1B-Instruct"
# MODEL_ID = "Qwen/Qwen3-1.7B"
# MODEL_ID = "stabilityai/stablelm-2-zephyr-1_6b"
# MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
# MODEL_ID = "Qwen/Qwen-1_8B-chat"
# MODEL_ID = "google/gemma-1.1-2b-it"
# MODEL_ID = "google/gemma-1.1-7b-it"
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                             trust_remote_code=True,
                                             dtype=torch.float16,
                                             device_map="cuda",
                                             quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                                                    bnb_4bit_compute_dtype=torch.float16))
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

refusal_dir = torch.load(MODEL_ID.replace("/", "_") + "_refusal_dir.pt")


def direction_ablation_hook(activation: jaxtyping.Float[torch.Tensor, "... d_act"],
                            direction: jaxtyping.Float[torch.Tensor, "d_act"]):
    proj = einops.einsum(activation, direction.view(-1, 1),
                         '... d_act, d_act single -> ... single') * direction
    return activation - proj


# Some model developers thought it was stupid to pass a tuple of tuple of tuples around (rightfully so), but unfortunately now we have a divide
sig = signature(model.model.layers[0].forward)
simple = sig.return_annotation == torch.Tensor


class AblationDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_type = "full_attention"

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ):
        assert not output_attentions

        ablated = direction_ablation_hook(hidden_states, refusal_dir.to(
            hidden_states.device)).to(hidden_states.device)

        if simple:
            return ablated

        outputs = (ablated,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


# for qwen 1 this needs to be changed to model.transformer.h
for idx in reversed(range(len(model.model.layers))):
    model.model.layers.insert(idx, AblationDecoderLayer())

# bruh
if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
    model.config.num_hidden_layers *= 2

conversation = []

streamer = TextStreamer(tokenizer)

print(f"Chat with {MODEL_ID}:")
while True:
    prompt = input()
    conversation.append({"role": "user", "content": prompt})
    toks = tokenizer.apply_chat_template(conversation=conversation,
                                         add_generation_prompt=True, return_tensors="pt")

    gen = model.generate(toks.to(model.device), streamer=streamer, max_new_tokens=1337)

    decoded = tokenizer.batch_decode(gen[0][len(toks[0]):], skip_special_tokens=True)
    conversation.append({"role": "assistant", "content": "".join(decoded)})
