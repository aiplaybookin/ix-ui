import math
import torch
from torch import nn
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from pytorch_lightning.loggers import TensorBoardLogger

# Dataset
class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, block_size=1024):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.tokens = self.tokenizer.encode(text)

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        input_ids = self.tokens[idx:idx + self.block_size]
        target_ids = self.tokens[idx + 1:idx + self.block_size + 1]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(target_ids, dtype=torch.long)
        }


# Lightning Module
class SmolLM2Lightning(pl.LightningModule):
    def __init__(self, vocab_size, block_size, lr, warmup_steps, max_steps, min_lr=0.0, pretrained_path=None):
        super().__init__()
        self.save_hyperparameters()

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
        config = LlamaConfig(**config_dict)
        self.model = LlamaForCausalLM(config)

        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if pretrained_path:
            pretrained_model = LlamaForCausalLM.from_pretrained(
                pretrained_path,
                torch_dtype=torch.bfloat16,
            )
            self.model.load_state_dict(pretrained_model.state_dict(), strict=True)
            print("Pretrained weights loaded successfully.")

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameter count: {param_count:,}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True)

        # Generate text to check coherence (only for first batch)
        if batch_idx == 0:
            with torch.no_grad():
                input_ids = batch["input_ids"]
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

                gen_outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                gen_text = self.tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
                print(f"\n[Step {self.global_step}] Generated: {gen_text}\n")

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01
        )

        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / self.hparams.warmup_steps
            elif current_step >= self.hparams.max_steps:
                return self.hparams.min_lr / self.hparams.lr if self.hparams.lr > 0 else 0.0
            elif current_step < 1600000:
                return 1.0
            else:
                decay_start = 1600000
                decay_steps = 400000
                progress = (current_step - decay_start) / decay_steps
                return 1.0 - progress

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def generate_text(self, prompt, max_new_tokens=100, temperature=0.7):
        """Generate text from a prompt for demo purposes."""
        self.model.eval()
        device = next(self.model.parameters()).device
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            gen_outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(gen_outputs[0], skip_special_tokens=True)

# Parameters
block_size = 1024
batch_size = 8
max_lr = 0.003
warmup_steps = 10
max_steps = 5000
min_lr = 0.0


# Tokenizer and vocab
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
vocab_size = 49152

# Dataset and DataLoader
dataset = TextDataset("input.txt", tokenizer, block_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# 🔥 SMALL VALIDATION SET so run completes in 2–5 mins
val_dataset = TextDataset("input.txt", tokenizer, block_size=128)
val_subset_size = min(2000, len(val_dataset))  # reduce for speed

val_dataset = Subset(val_dataset, list(range(val_subset_size)))

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# Set float32 matmul precision to high for better performance
torch.set_float32_matmul_precision('high')

# Logger
logger = TensorBoardLogger("logs/", name="smollm2_experiment")

# Progress bar
progress_bar = TQDMProgressBar(refresh_rate=1)

# Checkpoint callback - saves after training
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="smollm2-{step:05d}",
    save_top_k=-1,  # Save all checkpoints
    every_n_train_steps=500,  # Save every 500 steps
    save_last=True,  # Also save last checkpoint
)

# ====================================================================
# PHASE 0: Validating Model Architecture by loading Pretrained weights
# ====================================================================
print("\n" + "="*60)
print("PHASE 0: Validating Model Architecture by loading Pretrained weights")
print("="*60 + "\n")

pretrained_model = SmolLM2Lightning(
    vocab_size, 
    block_size, 
    max_lr, 
    warmup_steps, 
    max_steps, 
    min_lr=min_lr,
    pretrained_path="HuggingFaceTB/SmolLM2-135M"
)

print(pretrained_model)

# ============================================================
# PHASE 1: Train for 5000 steps with validation every 500 steps
# ============================================================
print("\n" + "="*60)
print("PHASE 1: Training for 5000 steps from scratch")
print("="*60 + "\n")

model = SmolLM2Lightning(
    vocab_size=vocab_size,
    block_size=block_size,
    lr=max_lr,
    warmup_steps=warmup_steps,
    max_steps=max_steps,
    min_lr=min_lr,
    pretrained_path=None  # THIS ensures fresh random initialization
)

trainer = pl.Trainer(
    max_steps=max_steps,
    accelerator="gpu",
    devices=1,
    callbacks=[
        LearningRateMonitor(logging_interval='step'), 
        progress_bar,
        checkpoint_callback
    ],
    precision='bf16-mixed',
    log_every_n_steps=1,
    enable_progress_bar=True,
    enable_model_summary=True,
    logger=logger,
    val_check_interval=500,  # Validate every 500 steps
)

trainer.fit(model, dataloader, val_dataloader)

# Save final checkpoint after Phase 1
final_ckpt_path = "checkpoints/smollm2-phase1-final.ckpt"
trainer.save_checkpoint(final_ckpt_path)
print(f"\n✅ Phase 1 complete! Checkpoint saved to: {final_ckpt_path}\n")

# ============================================================
# PHASE 2: Load checkpoint and train for 50 more steps
# ============================================================
print("\n" + "="*60)
print("PHASE 2: Loading checkpoint and training for 50 more steps")
print("="*60 + "\n")

# Load model from checkpoint
loaded_model = SmolLM2Lightning.load_from_checkpoint(
    final_ckpt_path,
    vocab_size=vocab_size,
    block_size=block_size,
    lr=max_lr,
    warmup_steps=warmup_steps,
    max_steps=max_steps + 50,  # Extend max_steps
    min_lr=min_lr,
)
print("✅ Checkpoint loaded successfully!")

# Checkpoint for phase 2
checkpoint_callback_phase2 = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="smollm2-phase2-{step:05d}",
    save_last=True,
)

trainer_phase2 = pl.Trainer(
    max_steps=50,  # Train for 50 more steps
    accelerator="gpu",
    devices=1,
    callbacks=[
        LearningRateMonitor(logging_interval='step'), 
        progress_bar,
        checkpoint_callback_phase2
    ],
    precision='bf16-mixed',
    log_every_n_steps=1,
    enable_progress_bar=True,
    enable_model_summary=False,
    logger=logger,
)

trainer_phase2.fit(loaded_model, dataloader)

# Save final checkpoint after Phase 2
final_ckpt_path_phase2 = "checkpoints/smollm2-phase2-final.ckpt"
trainer_phase2.save_checkpoint(final_ckpt_path_phase2)
print(f"\n✅ Phase 2 complete! Checkpoint saved to: {final_ckpt_path_phase2}\n")

# ============================================================
# PHASE 3: Load final model and generate demo text
# ============================================================
print("\n" + "="*60)
print("PHASE 3: Loading final model for demo generation")
print("="*60 + "\n")

# Load the final trained model
final_model = SmolLM2Lightning.load_from_checkpoint(
    final_ckpt_path_phase2,
    vocab_size=vocab_size,
    block_size=block_size,
    lr=max_lr,
    warmup_steps=warmup_steps,
    max_steps=max_steps + 50,
    min_lr=min_lr,
)
final_model = final_model.cuda()  # Move to GPU
final_model.eval()
print("✅ Final model loaded successfully!")

# Demo generation
demo_prompts = [
    "Once upon a time",
    "The meaning of life is",
    "In the beginning",
    "To be or not to be",
]

print("\n" + "-"*60)
print("DEMO GENERATIONS")
print("-"*60)

for prompt in demo_prompts:
    generated = final_model.generate_text(prompt, max_new_tokens=100, temperature=0.7)
    print(f"\n📝 Prompt: {prompt}")
    print(f"🤖 Generated: {generated}")
    print("-"*40)

print("\n✅ All phases complete!")