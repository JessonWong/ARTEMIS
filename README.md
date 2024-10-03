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
To get poisoned models, you can download them directly from [there](https://cuhko365.sharepoint.com/sites/SDSbackdoorbench/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FSDSbackdoorbench%2FShared%20Documents%2Fbackdoorbench&p=true&ga=1), or train with the script:
```python
bash attack.sh
```
To get full experiment results in our paper, directly run the script:
```
bash run.sh
```
