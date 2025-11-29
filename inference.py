"""Load trained model and generate text."""

import torch
from model import SmolLM2Lightning
from config import VOCAB_SIZE, TRAIN_CONFIG


def load_model(checkpoint_path: str):
    """Load model from checkpoint."""
    model = SmolLM2Lightning.load_from_checkpoint(
        checkpoint_path,
        vocab_size=VOCAB_SIZE,
        block_size=TRAIN_CONFIG["block_size"],
        lr=TRAIN_CONFIG["max_lr"],
        warmup_steps=TRAIN_CONFIG["warmup_steps"],
        max_steps=TRAIN_CONFIG["max_steps"],
        min_lr=TRAIN_CONFIG["min_lr"],
    )
    model = model.cuda()
    model.eval()
    print(f"✅ Model loaded from: {checkpoint_path}")
    return model


def generate_demo(model, prompts: list = None):
    """Generate text for demo prompts."""
    if prompts is None:
        prompts = [
            "Once upon a time",
            "The meaning of life is",
            "In the beginning",
            "To be or not to be",
        ]
    
    print("\n" + "-" * 60)
    print("DEMO GENERATIONS")
    print("-" * 60)
    
    for prompt in prompts:
        generated = model.generate_text(prompt, max_new_tokens=100, temperature=0.7)
        print(f"\n📝 Prompt: {prompt}")
        print(f"🤖 Generated: {generated}")
        print("-" * 40)


def main():
    # Load the final model
    checkpoint_path = "checkpoints/smollm2-phase2-final.ckpt"
    model = load_model(checkpoint_path)
    
    # Run demo
    generate_demo(model)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        prompt = input("\n📝 Enter prompt: ")
        if prompt.lower() == 'quit':
            break
        
        generated = model.generate_text(prompt, max_new_tokens=100, temperature=0.7)
        print(f"🤖 Generated: {generated}")
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()