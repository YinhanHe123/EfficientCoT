# SemCoT

## Introduction

This is the official code project for the paper "_SemCoT: Accelerating Chain-of-Thought Reasoning through Semantically-Aligned Implicit Tokens_."

**TL;DR**:  We make Chain-of-Thought reasoning in large language models (1) more efficient by creating implicit reasoning with lightweight language models; (2) still effective as the implicit reasoning maintains semantic alignment with ground-truth reasoning.

**Framework Overview**: The **SemCoT** (Semantically-aligned Implicit Chain-of-Thought) framework is a comprehensive approach to accelerating Chain-of-Thought reasoning in Large Language Models while preserving reasoning quality. It addresses two fundamental challenges in implicit CoT methods: maintaining semantic alignment between implicit and explicit reasoning, and optimizing token-level generation speed. The framework operates through a two-stage process. First, it trains a customized sentence transformer using contrastive learning to evaluate semantic alignment between implicit reasoning embeddings and ground-truth reasoning text. This transformer leverages the middle five layers of the target LLM as its backbone and employs pooling and linear projection layers to create semantic embeddings for similarity comparison. Second, SemCoT employs a lightweight language model (distilled/pruned from the original LLM) as an efficient implicit reasoning generator. This generator is trained with dual objectives: maximizing answer accuracy and maintaining semantic alignment with ground-truth reasoning (guided by the trained sentence transformer). The lightweight model generates special `<CoT>` tokens that are linearly projected into the LLM's embedding space to serve as implicit reasoning. By jointly optimizing semantic preservation and generation efficiency, SemCoT achieves superior performance in both effectiveness and speed compared to existing implicit CoT methods, making reasoning capabilities more  accessible for real-world applications.

![syncot_overview_textout-1](https://github.com/user-attachments/assets/d4f865f6-8ba2-4caf-81a0-f3806507b79e)

## Project Structure
```
project_root/
├── baselines/
│   ├── __init__.py
│   ├── baselines.py                    # Main baseline runner with method selection
│   ├── ccot_baseline_runner.py         # Compressed Chain of Thought baseline
│   ├── coconut_baseline_runner.py      # Chain of Continuous Thought baseline
│   ├── codi_baseline_runner.py         # CODI baseline implementation
│   ├── icot_kd_baseline_runner.py      # Implicit CoT with Knowledge Distillation
│   ├── icot_si_baseline_runner.py      # Implicit CoT Stepwise Internalization
│   ├── pause_baseline_runner.py        # Pause tokens baseline
│   └── softcot_baseline_runner.py      # SoftCoT baseline
│
├── config/
│   ├── __init__.py
│   ├── experiment_config.py            # Experiment configuration settings
│   └── model_config.py                 # Model configuration settings
│
├── data/
│   ├── __init__.py
│   ├── cot_datasets.py                 # Dataset loading and processing
│   ├── gpt4pair.py                     # GPT-4 reasoning pairs generation
│   └── reasoning_pairs.py              # Local reasoning pairs generation
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                      # Evaluation metrics and scoring
│
├── inference/
│   ├── __init__.py
│   └── inference.py                    # Inference pipeline for models
│
├── training/
│   ├── __init__.py
│   ├── loss_functions.py               # Custom loss functions (empty)
│   ├── train_ccot.py                   # CCoT model training
│   ├── train_coconut.py                # Coconut model training
│   ├── train_codi.py                   # CODI model training
│   ├── train_contemp_gen.py            # Contemplation generator training
│   ├── train_icot_kd.py                # Implicit CoT with KD training
│   ├── train_pause.py                  # Pause tokens model training
│   ├── train_sent_trans.py             # Sentence transformer training
│   └── train_softcot.py                # SoftCoT model training
│
├── utils/
│   ├── __init__.py
│   ├── logging.py                      # Logging utilities and TensorBoard
│   └── utils.py                        # General utility functions
│
└── main.py                             # Main entry point for running experiments
```


## Hyperparameter configs

### SVAMP
```bash
python main.py --config small --mode effi_cot --dataset svamp -stllr 0.0001 -stlwd 0.001 -stle 5 -stllmlr 1e-7 -stllmwd 1e-5 -stllme 2 -cgllr 0.0001 -cglwd 0.001 -cgle 5 -cgllmlr 1e-7 -cgllmwd 1e-5 -cgllme 2
```

```bash
python main.py --config mistral --mode effi_cot --dataset svamp --variation vanilla -stllr 0.01 -stlwd 0.0001 -stle 3 -stllmlr 1e-05 -stllmwd 1e-05 -stllme 2 -cgllr 0.0001 -cglwd 0.01 -cgle 3 -cgllmlr 1e-05 -cgllmwd 0.001 -cgllme 1
```

### Multiarith
```bash
python main.py --config small --mode effi_cot --dataset multiarith --variation vanilla -stllr 0.0001 -stlwd 0.001 -stle 5 -stllmlr 1e-07 -stllmwd 0.001 -stllme 1 -cgllr 0.001 -cglwd 0.01 -cgle 3 -cgllmlr 1e-05 -cgllmwd 0.001 -cgllme 2
```

```bash
python main.py --config mistral --mode effi_cot --dataset multiarith --variation vanilla -stllr 0.0001 -stlwd 0.01 -stle 1 -stllmlr 1e-07 -stllmwd 0.001 -stllme 2 -cgllr 0.001 -cglwd 0.0001 -cgle 5 -cgllmlr 1e-07 -cgllmwd 0.001 -cgllme 2
```

### Coin_flip
```bash
python main.py --config small --mode effi_cot --dataset coin_flip --variation vanilla -stllr 0.001 -stlwd 0.0001 -stle 3 -stllmlr 1e-05 -stllmwd 0.001 -stllme 1 -cgllr 0.01 -cglwd 0.01 -cgle 1 -cgllmlr 1e-07 -cgllmwd 0.001 -cgllme 2
```

```bash
python main.py --config mistral --mode effi_cot --dataset coin_flip --variation vanilla -stllr 0.0001 -stlwd 0.001 -stle 5 -stllmlr 1e-05 -stllmwd 0.001 -stllme 1 -cgllr 0.001 -cglwd 0.001 -cgle 3 -cgllmlr 1e-05 -cgllmwd 0.001 -cgllme 1
```

### GSM8K
```bash
python main.py --config small --mode effi_cot --dataset gsm8k --variation vanilla -stllr 0.001 -stlwd 0.01 -stle 3 -stllmlr 1e-05 -stllmwd 0.001 -stllme 1 -cgllr 0.0001 -cglwd 0.01 -cgle 3 -cgllmlr 1e-05 -cgllmwd 0.001 -cgllme 1
```

```bash
python main.py --config mistral --mode effi_cot --dataset gsm8k --variation vanilla -stllr 0.001 -stlwd 0.001 -stle 1 -stllmlr 1e-07 -stllmwd 1e-05 -stllme 1 -cgllr 0.01 -cglwd 0.01 -cgle 3 -cgllmlr 1e-07 -cgllmwd 0.001 -cgllme 2
```

### Commonsense_qa
```bash
python main.py --config small --mode effi_cot --dataset commonsense_qa --variation vanilla -stllr 0.0001 -stlwd 0.001 -stle 1 -stllmlr 1e-05 -stllmwd 1e-05 -stllme 2 -cgllr 0.01 -cglwd 0.01 -cgle 3 -cgllmlr 1e-05 -cgllmwd 0.001 -cgllme 1
```

```bash
python main.py --config mistral --mode effi_cot --dataset commonsense_qa --variation vanilla -stllr 0.01 -stlwd 0.001 -stle 1 -stllmlr 1e-05 -stllmwd 0.001 -stllme 1 -cgllr 0.01 -cglwd 0.01 -cgle 3 -cgllmlr 1e-05 -cgllmwd 0.001 -cgllme 1
```
