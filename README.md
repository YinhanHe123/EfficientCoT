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