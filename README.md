# SemCoT

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
## Run the code

### Step 1: Train the sentence transformer
python main.py --mode train_sentence_transformer --config default

### Step 2: Train the contemplation generator
python main.py --mode train_contemp_generator --config default

### Step 3: Evaluate the model
python main.py --mode evaluate --config default

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
