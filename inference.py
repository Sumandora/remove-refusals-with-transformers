from typing import Optional, Tuple

import einops
import jaxtyping
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig

torch.inference_mode()

torch.set_default_device("cpu")

# MODEL_ID = "stabilityai/stablelm-2-1_6b"
# MODEL_ID = "stabilityai/stablelm-2-zephyr-1_6b"
# MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
# MODEL_ID = "Qwen/Qwen-1_8B-chat"
MODEL_ID = "google/gemma-1.1-2b-it"
# MODEL_ID = "google/gemma-1.1-7b-it"
# MODEL_ID = "unsloth/gemma-1.1-7b-it-bnb-4bit"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="auto", quantization_config=BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True, bnb_4bit_compute_dtype=torch.float16))
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

refusal_dir = torch.load(MODEL_ID.replace("/", "_") + "_refusal_dir.pt")


def direction_ablation_hook(activation: jaxtyping.Float[torch.Tensor, "... d_act"],
                            direction: jaxtyping.Float[torch.Tensor, "d_act"]):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj


class AblationDecoderLayer(nn.Module):
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
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        assert not output_attentions

        ablated = direction_ablation_hook(hidden_states, refusal_dir.to(hidden_states.device)).to(hidden_states.device)

        outputs = (ablated,)

        if use_cache:
            outputs += (past_key_value,)

        # noinspection PyTypeChecker
        return outputs


for idx in reversed(range(len(model.model.layers))):  # for qwen 1 this needs to be changed to model.transformer.h
    model.model.layers.insert(idx, AblationDecoderLayer())

conversation=[]

streamer = TextStreamer(tokenizer)

while True:
    prompt = input()
    conversation.append({"role": "user", "content": prompt})
    toks = tokenizer.apply_chat_template(conversation=conversation,
        add_generation_prompt=True, return_tensors="pt")

    gen = model.generate(toks.to(model.device), streamer=streamer, max_new_tokens=1337)

    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
    conversation.append({"role": "assistant", "content": decoded})

