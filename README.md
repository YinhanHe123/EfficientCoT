# EfficientCoT

## Project Structure
```
contemplation-reasoning/
├── main.py                    # Entry point for training and evaluation
├── config/
│   ├── __init__.py
│   ├── model_config.py        # Configuration for models, paths, hyperparameters
│   └── experiment_config.py   # Configuration for experiments and evaluation
├── data/
│   ├── __init__.py
│   ├── datasets.py            # Dataset loading and processing (raw)
│   └── reasoning_pairs.py     # Generate original and condensed reasoning pairs
├── models/
│   ├── __init__.py
│   ├── contemp_generator.py   # Student model that generates contemplation tokens
│   ├── sentence_transformer.py # Customized sentence transformer
│   └── utils.py               # Model utilities
├── training/
│   ├── __init__.py
│   ├── train_contemp_gen.py   # Training loop for contemplation generator
│   ├── train_sent_trans.py    # Training for customized sentence transformer
│   └── loss_functions.py      # Implementation of Lreason and Lans
├── inference/
│   ├── __init__.py
│   └── inference.py           # Inference pipeline
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # Evaluation metrics
│   └── baselines.py           # Implementation of baseline methods (CCOT, Pause, Implicit CoT)
└── utils/
    ├── __init__.py
    ├── hidden_states.py       # Utilities for handling hidden states
    └── logging.py             # Logging utilities
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