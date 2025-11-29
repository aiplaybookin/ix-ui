"""Configuration and hyperparameters."""

# Model architecture
MODEL_CONFIG = {
    "bos_token_id": 0,
    "eos_token_id": 0,
    "hidden_act": "silu",
    "hidden_size": 576,
    "initializer_range": 0.041666666666666664,
    "intermediate_size": 1536,
    "max_position_embeddings": 2048,
    "num_attention_heads": 9,
    "num_hidden_layers": 30,
    "num_key_value_heads": 3,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "use_cache": True,
}

# Training hyperparameters
TRAIN_CONFIG = {
    "block_size": 1024,
    "batch_size": 8,
    "max_lr": 0.003,
    "min_lr": 0.0,
    "warmup_steps": 10,
    "max_steps": 5000,
    "val_check_interval": 500,
    "save_every_n_steps": 500,
}

# Data settings
DATA_CONFIG = {
    "train_file": "input.txt",
    "val_block_size": 128,
    "val_subset_size": 2000,
    "num_workers": 8,
    "val_num_workers": 2,
}

# Tokenizer
TOKENIZER_NAME = "HuggingFaceTB/cosmo2-tokenizer"
VOCAB_SIZE = 49152

# Pretrained model (set to None for training from scratch)
PRETRAINED_PATH = "HuggingFaceTB/SmolLM2-135M"

# Paths
CHECKPOINT_DIR = "checkpoints/"
LOG_DIR = "logs/"
EXPERIMENT_NAME = "smollm2_experiment"

# Hugging Face upload settings
HF_CONFIG = {
    "username": "vikashkr117",
    "model_name": "smollm2-135m-raw-trained",
    "private": False,
}