class ModelConfig:
    def __init__(self, config_name="small"):
        self.config_name = config_name

        # Model names
        self.teacher_model_name = "meta-llama/Llama-2-7b-hf"
        self.student_model_name = "princeton-nlp/Sheared-LLaMA-1.3B"

        # Model paths for loading/saving
        self.student_model_path = "./saved_models/contemp_generator"
        self.sentence_transformer_path = "./saved_models/sentence_transformer"

        # Data paths
        self.data_path = "openai/gsm8k"

        # Model dimensions
        self.teacher_hidden_dim = 4096  # Hidden dimension of the teacher model
        self.contemp_seq_length = 64  # Length of contemplation token sequence

        # Layer indices
        self.contemp_layer_index = 16  # Layer to extract hidden states from

        # Load config-specific settings
        self._load_specific_config()

    def _load_specific_config(self):
        if self.config_name == "small":
            self.teacher_model_name = "meta-llama/Llama-2-7b-chat-hf"
            self.student_model_name = "princeton-nlp/Sheared-LLaMA-1.3B"
            self.teacher_hidden_dim = 4096
            self.contemp_seq_length = 32
        elif self.config_name == "large":
            self.teacher_model_name = "meta-llama/Llama-2-13b-hf"
            self.student_model_name = "meta-llama/Llama-2-7b-hf"
            self.teacher_hidden_dim = 2000
            self.contemp_seq_length = 128