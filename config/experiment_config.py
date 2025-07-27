import os
import torch

class ExperimentConfig:
    def __init__(self, config_name="default"):
        self.config_name = config_name

        # Paths
        self.log_dir = "./logs"
        self.model_save_path = "/data/nee7ne/effi_cot2/saved_models"
        self.checkpoint_path = "./checkpoints"
        self.result_path = "./results"
        self.experiment_name = "default_experiment"

        # Training parameters for sentence transformer
        self.sent_trans_lr = 1e-5
        self.sent_trans_weight_decay = 0.01
        self.sent_trans_epochs = 15

        # Add missing sentence transformer parameters
        self.st_linear_lr = 0.001
        self.st_linear_wd = 0.001
        self.st_linear_epochs = 5
        self.st_llm_lr = 1e-5
        self.st_llm_wd = 0.01
        self.st_llm_epochs = 10

        # Training parameters for contemplation generator
        self.contemp_gen_lr = 1e-7
        self.contemp_gen_weight_decay = 1e-5
        self.contemp_gen_epochs = 2

        self.contemp_gen_lin_layer_lr = 0.001
        self.contemp_gen_lin_layer_weight_decay = 0.001
        self.contemp_gen_lin_layer_epochs = 10

        # Add missing contemplation generator parameters (cg prefix)
        self.cg_linear_lr = 0.001
        self.cg_linear_wd = 0.001
        self.cg_linear_epochs = 10
        self.cg_llm_lr = 1e-7
        self.cg_llm_wd = 1e-5
        self.cg_llm_epochs = 2

        self.batch_size = 4
        self.alpha = 0.25  # Weight for Lreason in total loss
        self.save_interval = 1  # Save model every N epochs
        self.max_seq_length = 512
        self.embedding_dim = 768  # Dimension for embeddings
        self.seed = 42
        # self.max_reasoning_pairs = 1000  # Maximum reasoning pairs to generate
        # self.max_reasoning_pairs = 7473
        self.max_reasoning_pairs = 800
        self.train_max_contemp_tokens = 5
        self.eval_max_contemp_tokens = 1

        # Model-specific parameters
        self.start_layer_idx = 16  # Start layer for sentence transformer
        self.end_layer_idx = 20  # End layer for sentence transformer
        directory = str(os.path.abspath(os.path.join(__file__ ,"../..")))
        self.reasoning_pairs_path = os.path.join(directory, "gen_datasets")

        self.device = None
        self.ccot_stage = None
        self.ccot_lr = 1e-5

        self.eval_temp = 0.7

        self.codi_lr = 8e-4
        # Load config-specific settings
        # self._load_specific_config()

        self.coconut_stage = None