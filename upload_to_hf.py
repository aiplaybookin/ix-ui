"""Upload trained model to Hugging Face Hub."""

import os
import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo

from config import VOCAB_SIZE, TOKENIZER_NAME, HF_CONFIG, TRAIN_CONFIG
from model import SmolLM2Lightning


def load_checkpoint_weights(checkpoint_path: str):
    """Load weights from Lightning checkpoint."""
    # Load the full Lightning module
    lightning_model = SmolLM2Lightning.load_from_checkpoint(
        checkpoint_path,
        vocab_size=VOCAB_SIZE,
        block_size=TRAIN_CONFIG["block_size"],
        lr=TRAIN_CONFIG["max_lr"],
        warmup_steps=TRAIN_CONFIG["warmup_steps"],
        max_steps=TRAIN_CONFIG["max_steps"],
        min_lr=TRAIN_CONFIG["min_lr"],
    )
    
    # Extract the inner HuggingFace model
    model = lightning_model.model
    print(f"✅ Weights loaded from: {checkpoint_path}")
    
    return model


def create_model_card(repo_id: str):
    """Create README for the model."""
    return f"""---
license: apache-2.0
base_model: HuggingFaceTB/SmolLM2-135M
tags:
  - smollm2
  - pytorch
  - causal-lm
  - fine-tuned
---

# {HF_CONFIG['model_name']}

This model is a fine-tuned version of [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

## Model Details

- **Base Model:** SmolLM2-135M
- **Parameters:** ~135M
- **Training:** Fine-tuned using PyTorch Lightning

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")

# Generate text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""


def upload_model(checkpoint_path: str):
    """Upload model to Hugging Face Hub."""
    repo_id = f"{HF_CONFIG['username']}/{HF_CONFIG['model_name']}"
    local_dir = f"./hf_model/{HF_CONFIG['model_name']}"
    
    print(f"Preparing upload to: {repo_id}")
    
    # Load model (HuggingFace model extracted from Lightning)
    model = load_checkpoint_weights(checkpoint_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Save locally
    os.makedirs(local_dir, exist_ok=True)
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    print(f"✅ Model saved locally to: {local_dir}")
    
    # Create model card
    with open(f"{local_dir}/README.md", "w") as f:
        f.write(create_model_card(repo_id))
    print("✅ Model card created!")
    
    # Upload to Hub
    api = HfApi()
    
    try:
        create_repo(repo_id=repo_id, private=HF_CONFIG["private"], exist_ok=True)
        print(f"✅ Repository created: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        commit_message="Upload fine-tuned SmolLM2-135M model"
    )
    
    print(f"\n🎉 Upload complete!")
    print(f"🔗 View your model at: https://huggingface.co/{repo_id}")


def main():
    checkpoint_path = "checkpoints/smollm2-phase2-final.ckpt"
    upload_model(checkpoint_path)


if __name__ == "__main__":
    main()