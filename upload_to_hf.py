import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from huggingface_hub import HfApi, create_repo
import pytorch_lightning as pl
import os

# ============================================================
# CONFIGURATION - Update these values
# ============================================================
CHECKPOINT_PATH = "checkpoints/smollm2-phase2-final.ckpt"  # Your checkpoint
HF_USERNAME = "your-username"  # Your Hugging Face username
MODEL_NAME = "smollm2-135m-finetuned"  # Name for your model
PRIVATE = False  # Set True for private repo

# Full repo ID
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

# ============================================================
# Load your Lightning checkpoint
# ============================================================
print("Loading checkpoint...")

# Model config (same as training)
vocab_size = 49152
config_dict = {
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
    "vocab_size": vocab_size,
}

# Create model with config
config = LlamaConfig(**config_dict)
model = LlamaForCausalLM(config)

# Load weights from checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
state_dict = checkpoint["state_dict"]

# Remove "model." prefix from keys (Lightning adds this)
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("model.", "") if key.startswith("model.") else key
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
print("✅ Checkpoint loaded!")

# ============================================================
# Load tokenizer
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# Save model locally first
# ============================================================
LOCAL_DIR = f"./hf_model/{MODEL_NAME}"
os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"Saving model to {LOCAL_DIR}...")
model.save_pretrained(LOCAL_DIR)
tokenizer.save_pretrained(LOCAL_DIR)
print("✅ Model saved locally!")

# ============================================================
# Create model card (README.md)
# ============================================================
model_card = f"""---
license: apache-2.0
base_model: HuggingFaceTB/SmolLM2-135M
tags:
  - smollm2
  - pytorch
  - causal-lm
  - fine-tuned
---

# {MODEL_NAME}

This model is a fine-tuned version of [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

## Model Details

- **Base Model:** SmolLM2-135M
- **Parameters:** ~135M
- **Training:** Fine-tuned using PyTorch Lightning

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{REPO_ID}")
model = AutoModelForCausalLM.from_pretrained("{REPO_ID}")

# Generate text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

- Framework: PyTorch Lightning
- Precision: bf16-mixed
- Optimizer: AdamW
"""

with open(f"{LOCAL_DIR}/README.md", "w") as f:
    f.write(model_card)
print("✅ Model card created!")

# ============================================================
# Upload to Hugging Face Hub
# ============================================================
print(f"\nUploading to Hugging Face: {REPO_ID}...")

api = HfApi()

# Create repo (if it doesn't exist)
try:
    create_repo(repo_id=REPO_ID, private=PRIVATE, exist_ok=True)
    print(f"✅ Repository created: https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"Repository might already exist: {e}")

# Upload all files
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    commit_message="Upload fine-tuned SmolLM2-135M model"
)

print(f"\n🎉 Upload complete!")
print(f"🔗 View your model at: https://huggingface.co/{REPO_ID}")