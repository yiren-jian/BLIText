# Bootstrapping Vision-Language Learning with Decoupled Language Pre-training


This repo covers implementations of BLIP2 + Pformer in **Bootstrapping Vision-Language Learning with Decoupled Language Pre-training** by [Yiren Jian](https://yiren-jian.github.io/), [Chongyang Gao](https://gcyzsl.github.io/) and [Soroush Vosoughi](https://www.cs.dartmouth.edu/~soroush/).

<img src="overview.png" width="800">

The code is developed based on [LAVIS](https://github.com/salesforce/LAVIS/) project (cloned on Feb 23, 2023).

We mainly add following files in `lavis/models/blip2_models`:

- [x] `blip2_pformer_opt.py`
- [x] `blip2_pformer_t5.py`
- [x] `blip2_pformer.py`
- [x] `pformer_opt.py`
- [x] `pformer_t5.py`

We also edit `lavis/models/base_model.py` to allow training from scratch, and include new `dataset` and `dataloader` in `lavis/datasets` of pure sentence dataset for training P-Former.

## Installation

```bash
conda create -n lavis python=3.8
conda activate lavis
pip install -e .
```

## Data Preparation
Please follow instructions from [LAVIS](https://github.com/salesforce/LAVIS/) to download pre-training datasets.

## Training
The code will be released soon.

## Evaluation
The code will be released soon.
