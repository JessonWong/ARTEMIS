# ARTEMIS
Code release for [ARTEMIS: Defending against Backdoor Attacks via Distribution Shift](https://arxiv.org/abs/2402.12343).


This repo includes:
- A new approach for defending backdoor attacks inspired by domain distribution shift, which can effectively purify poisoned models with minimum decrease on performance on benigned samples, especially for defending low-poisoning-rate, it outperforms current defending methods.
- The official implementation of current backdoor attacks and defenses based on [BackdoorBench](https://github.com/SCLBD/BackdoorBench) for comparing performance across different attack and defense methods.


## Installation

```bash
conda env create -f cfg.yaml
```


## Experiments
To reproduce our results in paper, run following commands:
```python
from inference_time_alignment.decoder import EFTPosthocGenerationMixin

chat_7b_model  = ... # Llama-2-7b-chat
base_7b_model  = ... # Llama-2-7b
base_13b_model = ... # Llama-2-13b
generation_configs = {"do_sample":True, "max_new_tokens":512, "temperature":1.0}

# logp_{eft} = logp_{base,13b} + 1.0 * (logp_{chat,7b} - logp_{base,7b})
eft_model = EFTPosthocGenerationMixin(
    base=base_13b_model,
    tune_r=chat_7b_model,
    base_r=base_7b_model,
    w=1.0,
)

# use transformer generate api as is
tokenized_output = eft_model.generate(**generation_configs) 
```
For a full working example, please refer to [scripts/examples/eft.py](https://github.com/ZHZisZZ/emulated-disalignment/tree/main/scripts/examples/eft.py).


## ED example
Our ED implementation is based on our EFT implementation. Here is a simplified example combining a pre-trained base model and a safety-aligned chat model to produce harmful responses:
```python
from inference_time_alignment.decoder import EFTPosthocGenerationMixin

chat_7b_model  = ... # Llama-2-7b-chat
base_7b_model  = ... # Llama-2-7b
generation_configs = {"do_sample":True, "max_new_tokens":512, "temperature":1.0}
alpha = 0.3

# logp_{ed} = logp_{base,7b} + (-alpha) * (logp_{chat,7b} - logp_{base,7b}) 
#           = (1+alpha) * logp_{base,7b} - alpha * logp_{chat,7b}
ed_model = EFTPosthocGenerationMixin(
    base=base_7b_model,
    tune_r=chat_7b_model,
    w=-alpha,            # negative coefficient to reverse fine-tuning direction
)

# use transformer generate api as is
tokenized_output = ed_model.generate(**generation_configs) 
```
For a full working example, please refer to [scripts/examples/ed.py](https://github.com/ZHZisZZ/emulated-disalignment/tree/main/scripts/examples/ed.py).


## ED interactive demo
Run `python ed_demo.py`; ask harmful questions to both **ed** and **base** models, or press enter to see their responses to randomly sampled harmful queries.

By default, this demo attacks [Alpaca](https://huggingface.co/PKU-Alignment/alpaca-7b-reproduced)x[Beaver](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0) with [Llama-Guard](https://huggingface.co/meta-llama/LlamaGuard-7b) as evaluator.
To attack other model pairs, e.g., [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf)x[Llama-2-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf):

```bash
python ed_demo.py --family_name llama-2
```

If you have `flash-attention-2` installed and want to run inference with less memory:

```bash
python ed_demo.py --use-flash-attention-2 --load-in-4bit
```

<details>
<summary>Click for detailed args</summary>

```bash
usage: ed_demo.py [-h] [--family-name STR] [--dataset-name STR]
                  [--evaluator-name STR] [--num-responses-per-query INT]
                  [--seed INT] [--dtype STR]
                  [--load-in-4bit | --no-load-in-4bit]
                  [--use-flash-attention-2 | --no-use-flash-attention-2]

╭─ arguments ────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                    │
│ --family-name STR       `llama-2`, `llama`, `mistral` or `alpaca`          │
│                         (default: alpaca)                                  │
│ --dataset-name STR      `Anthropic/hh-rlhf`, `lmsys/toxic-chat`,           │
│                         `mmathys/openai-moderation-api-evaluation` or      │
│                         `PKU-Alignment/BeaverTails` (default:              │
│                         PKU-Alignment/BeaverTails)                         │
│ --evaluator-name STR    `llama-guard` or `openai-moderation` (default:     │
│                         llama-guard)                                       │
│ --num-responses-per-query INT                                              │
│                         number of responses for each query (default: 3)    │
│ --seed INT              (default: 0)                                       │
│ --dtype STR             `bfloat16` or `float16` (default: bfloat16)        │
│ --load-in-4bit, --no-load-in-4bit                                          │
│                         True if OOM encountered (default: False)           │
│ --use-flash-attention-2, --no-use-flash-attention-2                        │
│                         True to use flash attention 2                      │
│                         (default: False)                                   │
╰────────────────────────────────────────────────────────────────────────────╯
```

</details>

## Reference

```
@article{zhou2024emulated,
  title={Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!},
  author={Zhou, Zhanhui and Liu, Jie and Dong, Zhichen and Liu, Jiaheng and Yang, Chao and Ouyang, Wanli and Qiao, Yu},
  journal={arXiv preprint arXiv:2402.12343},
  year={2024}
}
```
