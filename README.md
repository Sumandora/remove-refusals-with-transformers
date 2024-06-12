# Removing refusals with HF Transformers

This is a crude, proof-of-concept implementation to remove refusals from an LLM model without using TransformerLens. This means, that this supports every model that HF Transformers supports*.

The code was tested on a RTX 2060 6GB, thus mostly <3B models have been tested, but the code has been tested to work with bigger models as well.

*While most models are compatible, some models are not. Mainly because of custom model implementations. Some Qwen implementations for example don't work. Because `model.model.layers` can't be used for getting layers. They call the variables so that, `model.transformer.h` must be used, if I'm not mistaken.

## Usage
1. Set model and quantization in compute_refusal_dir.py and inference.py (Quantization can apparently be mixed)
2. Run compute_refusal_dir.py (Some settings in that file may be changed depending on your use-case)
3. Run inference.py and ask the model how to build an army of rabbits, that will overthrow your local government one day, by stealing all the carrots.

## Credits
- [Harmful instructions](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv)
- [Harmless instructions](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- [Technique](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)
