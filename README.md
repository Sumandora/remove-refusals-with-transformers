# Removing refusals with HF Transformers

This is a crude, proof-of-concept implementation to remove refusals from an LLM model without using TransformerLens. This means that this supports every model that HF Transformers supports*.

*While most models are compatible, some models are not. Mainly because of custom implementations. Some Qwen implementations for example don't work with ˋmodel.model.layersˋ for getting layers. They need ˋmodel.transformer.hˋ if I'm not mistaken.

## Usage
1. Set model, quantization in compute_refusal_dir.py and inference.py (In my testing, quantization can be mixed)
2. Set prompt in inference.py
3. First compute_refusal_dir.py
4. Run inference.py

## Credits
- [Harmful instructions](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv)
- [Harmless instructions](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- [Technique](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)
