"""Main training script."""

import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from config import (
    TRAIN_CONFIG, DATA_CONFIG, VOCAB_SIZE, TOKENIZER_NAME,
    PRETRAINED_PATH, CHECKPOINT_DIR, LOG_DIR, EXPERIMENT_NAME
)
from dataset import TextDataset
from model import SmolLM2Lightning


def create_dataloaders(tokenizer):
    """Create train and validation dataloaders."""
    # Training dataset
    train_dataset = TextDataset(
        DATA_CONFIG["train_file"], 
        tokenizer, 
        TRAIN_CONFIG["block_size"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=DATA_CONFIG["num_workers"],
        pin_memory=True
    )
    
    # Validation dataset (smaller)
    val_dataset = TextDataset(
        DATA_CONFIG["train_file"],
        tokenizer,
        block_size=DATA_CONFIG["val_block_size"]
    )
    val_subset_size = min(DATA_CONFIG["val_subset_size"], len(val_dataset))
    val_dataset = Subset(val_dataset, list(range(val_subset_size)))
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=DATA_CONFIG["val_num_workers"],
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_callbacks():
    """Create training callbacks."""
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=1),
        ModelCheckpoint(
            dirpath=CHECKPOINT_DIR,
            filename="smollm2-{step:05d}",
            save_top_k=-1,
            every_n_train_steps=TRAIN_CONFIG["save_every_n_steps"],
            save_last=True,
        ),
    ]
    return callbacks


def train_phase1(model, train_loader, val_loader, logger):
    """Phase 1: Main training loop."""
    print("\n" + "=" * 60)
    print("PHASE 1: Training for {} steps".format(TRAIN_CONFIG["max_steps"]))
    print("=" * 60 + "\n")
    
    trainer = pl.Trainer(
        max_steps=TRAIN_CONFIG["max_steps"],
        accelerator="gpu",
        devices=1,
        callbacks=create_callbacks(),
        precision='bf16-mixed',
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=logger,
        val_check_interval=TRAIN_CONFIG["val_check_interval"],
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    # Save final checkpoint
    final_ckpt = f"{CHECKPOINT_DIR}/smollm2-phase1-final.ckpt"
    trainer.save_checkpoint(final_ckpt)
    print(f"\n✅ Phase 1 complete! Checkpoint saved to: {final_ckpt}\n")
    
    return final_ckpt


def train_phase2(checkpoint_path, train_loader, logger, extra_steps=50):
    """Phase 2: Continue training from checkpoint."""
    print("\n" + "=" * 60)
    print(f"PHASE 2: Loading checkpoint and training for {extra_steps} more steps")
    print("=" * 60 + "\n")
    
    # Load from checkpoint
    model = SmolLM2Lightning.load_from_checkpoint(
        checkpoint_path,
        vocab_size=VOCAB_SIZE,
        block_size=TRAIN_CONFIG["block_size"],
        lr=TRAIN_CONFIG["max_lr"],
        warmup_steps=TRAIN_CONFIG["warmup_steps"],
        max_steps=TRAIN_CONFIG["max_steps"] + extra_steps,
        min_lr=TRAIN_CONFIG["min_lr"],
    )
    print("✅ Checkpoint loaded successfully!")
    
    # Phase 2 checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="smollm2-phase2-{step:05d}",
        save_last=True,
    )
    
    trainer = pl.Trainer(
        max_steps=extra_steps,
        accelerator="gpu",
        devices=1,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            TQDMProgressBar(refresh_rate=1),
            checkpoint_callback
        ],
        precision='bf16-mixed',
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=logger,
    )
    
    trainer.fit(model, train_loader)
    
    # Save final checkpoint
    final_ckpt = f"{CHECKPOINT_DIR}/smollm2-phase2-final.ckpt"
    trainer.save_checkpoint(final_ckpt)
    print(f"\n✅ Phase 2 complete! Checkpoint saved to: {final_ckpt}\n")
    
    return final_ckpt


def main():
    # Set precision
    torch.set_float32_matmul_precision('high')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(tokenizer)
    
    # Create logger
    logger = TensorBoardLogger(LOG_DIR, name=EXPERIMENT_NAME)
    
    # ============================================================
    # PHASE 0: Validate architecture with pretrained weights
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 0: Validating Model Architecture")
    print("=" * 60 + "\n")
    
    pretrained_model = SmolLM2Lightning(
        vocab_size=VOCAB_SIZE,
        block_size=TRAIN_CONFIG["block_size"],
        lr=TRAIN_CONFIG["max_lr"],
        warmup_steps=TRAIN_CONFIG["warmup_steps"],
        max_steps=TRAIN_CONFIG["max_steps"],
        min_lr=TRAIN_CONFIG["min_lr"],
        pretrained_path=PRETRAINED_PATH
    )
    print(pretrained_model)
    
    # ============================================================
    # PHASE 1: Train from scratch
    # ============================================================
    model = SmolLM2Lightning(
        vocab_size=VOCAB_SIZE,
        block_size=TRAIN_CONFIG["block_size"],
        lr=TRAIN_CONFIG["max_lr"],
        warmup_steps=TRAIN_CONFIG["warmup_steps"],
        max_steps=TRAIN_CONFIG["max_steps"],
        min_lr=TRAIN_CONFIG["min_lr"],
        pretrained_path=None  # Train from scratch
    )
    
    phase1_ckpt = train_phase1(model, train_loader, val_loader, logger)
    
    # ============================================================
    # PHASE 2: Continue training
    # ============================================================
    phase2_ckpt = train_phase2(phase1_ckpt, train_loader, logger, extra_steps=50)
    
    print("\n✅ All training phases complete!")
    print(f"Final checkpoint: {phase2_ckpt}")


if __name__ == "__main__":
    main()