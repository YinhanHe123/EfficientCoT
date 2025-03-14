import os
import torch

class ExperimentConfig:
    def __init__(self, config_name="default"):
        self.config_name = config_name

        # Paths
        self.log_dir = "./logs"
        self.model_save_path = "/data/nee7ne/effi_cot/saved_models"
        self.checkpoint_path = "./checkpoints"
        self.result_path = "./results"
        self.experiment_name = "default_experiment"

        # Training parameters
        self.learning_rate = 1e-5
        self.weight_decay = 0.01
        self.num_epochs = 5 # for debugging
        self.train_sen_trans_epochs = 15
        self.batch_size = 4
        self.alpha = 0.5  # Weight for Lreason in total loss
        self.save_interval = 1  # Save model every N epochs
        self.max_seq_length = 512
        self.embedding_dim = 768  # Dimension for embeddings
        # self.max_reasoning_pairs = 1000  # Maximum reasoning pairs to generate
        # self.max_reasoning_pairs = 7473
        self.max_reasoning_pairs = 16 # for debugging
        self.max_contemp_tokens = 15

        # Model-specific parameters
        self.start_layer_idx = 16  # Start layer for sentence transformer
        self.end_layer_idx = 20  # End layer for sentence transformer

        directory = str(os.path.abspath(os.path.join(__file__ ,"../..")))
        self.reasoning_pairs_path = os.path.join(directory, "gen_datasets")


        self.device = None

        # Load config-specific settings
        # self._load_specific_config()