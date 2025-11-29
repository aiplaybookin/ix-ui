"""Main training script."""

import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer
from pytorch_lightning import seed_everything

from config import (
    TRAIN_CONFIG, DATA_CONFIG, VOCAB_SIZE, TOKENIZER_NAME,
    PRETRAINED_PATH, CHECKPOINT_DIR, LOG_DIR, EXPERIMENT_NAME, SEED,
    calculate_sample_offset, calculate_tokens_processed
)
from dataset import TextDataset
from model import SmolLM2Lightning


def create_dataloaders(tokenizer, sample_offset: int = 0):
    """
    Create train and validation dataloaders.
    
    Args:
        tokenizer: HuggingFace tokenizer
        sample_offset: Starting token position (sample index = token position)
    """
    # Training dataset
    train_dataset = TextDataset(
        DATA_CONFIG["train_file"], 
        tokenizer, 
        TRAIN_CONFIG["block_size"],
        token_offset=sample_offset  # This is the starting token position
    )
    
    if len(train_dataset) == 0:
        raise ValueError(f"No samples remaining after offset {sample_offset}!")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=DATA_CONFIG["num_workers"],
        pin_memory=True
    )
    
    # Validation dataset (no offset - validate on full data)
    val_dataset = TextDataset(
        DATA_CONFIG["train_file"],
        tokenizer,
        block_size=DATA_CONFIG["val_block_size"],
        token_offset=0  # Validation uses full data
    )
    val_subset_size = min(DATA_CONFIG["val_subset_size"], len(val_dataset))
    
    if val_subset_size > 0:
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


def get_dataset_info(tokenizer):
    """Get dataset statistics."""
    with open(DATA_CONFIG["train_file"], 'r') as f:
        text = f.read()
    total_tokens = len(tokenizer.encode(text))
    total_samples = total_tokens - TRAIN_CONFIG["block_size"]
    steps_per_epoch = total_samples // TRAIN_CONFIG["batch_size"]
    
    return {
        "total_tokens": total_tokens,
        "total_samples": total_samples,
        "steps_per_epoch": steps_per_epoch,
    }


def train_phase1(model, train_loader, val_loader, logger, dataset_info):
    """Phase 1: Main training loop."""
    print("\n" + "=" * 60)
    print("PHASE 1: Training for {} steps".format(TRAIN_CONFIG["max_steps"]))
    print("=" * 60 + "\n")
    
    epochs = TRAIN_CONFIG["max_steps"] / dataset_info["steps_per_epoch"]
    samples_to_process = TRAIN_CONFIG["max_steps"] * TRAIN_CONFIG["batch_size"]
    
    print(f"Dataset info:")
    print(f"  Total tokens:      {dataset_info['total_tokens']:,}")
    print(f"  Total samples:     {dataset_info['total_samples']:,}")
    print(f"  Steps per epoch:   {dataset_info['steps_per_epoch']:,}")
    print(f"Training plan:")
    print(f"  Steps:             {TRAIN_CONFIG['max_steps']:,}")
    print(f"  Samples to use:    {samples_to_process:,}")
    print(f"  Epochs:            {epochs:.2f}")
    print()
    
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
    
    print(f"\n✅ Phase 1 complete!")
    print(f"Checkpoint saved to: {final_ckpt}")
    print(f"Steps completed: {TRAIN_CONFIG['max_steps']:,}")
    print(f"Samples processed: {samples_to_process:,}")
    print(f"Epochs completed: {epochs:.2f}\n")
    
    return final_ckpt, TRAIN_CONFIG["max_steps"]


def train_phase2(checkpoint_path, tokenizer, logger, dataset_info, completed_steps: int, extra_steps: int = 50):
    """Phase 2: Continue training from checkpoint with remaining data."""
    print("\n" + "=" * 60)
    print(f"PHASE 2: Loading checkpoint and training for {extra_steps} more steps")
    print("=" * 60 + "\n")
    
    # Calculate CORRECT sample offset
    sample_offset = calculate_sample_offset(completed_steps)
    
    print(f"Phase 1 stats:")
    print(f"  Completed steps:     {completed_steps:,}")
    print(f"  Samples processed:   {sample_offset:,}")
    print(f"Dataset stats:")
    print(f"  Total tokens:        {dataset_info['total_tokens']:,}")
    print(f"  Total samples:       {dataset_info['total_samples']:,}")
    print(f"Phase 2 plan:")
    print(f"  Starting from token: {sample_offset:,}")
    print(f"  Remaining tokens:    {dataset_info['total_tokens'] - sample_offset:,}")
    print(f"  Remaining samples:   {dataset_info['total_samples'] - sample_offset:,}")
    print()
    
    # Check if we have enough data
    remaining_samples = dataset_info['total_samples'] - sample_offset
    if remaining_samples <= 0:
        print(f"⚠️  No data remaining! Already processed all {dataset_info['total_samples']:,} samples.")
        print(f"To continue training, either:")
        print(f"  1. Use a larger dataset")
        print(f"  2. Reset offset to 0 (train another epoch on same data)")
        return checkpoint_path
    
    samples_needed = extra_steps * TRAIN_CONFIG["batch_size"]
    if remaining_samples < samples_needed:
        print(f"⚠️  Warning: Only {remaining_samples:,} samples remaining, but {samples_needed:,} needed.")
        print(f"Will train on available data.\n")
    
    # Create dataloaders with CORRECT offset
    train_loader, val_loader = create_dataloaders(tokenizer, sample_offset=sample_offset)
    
    # Load from checkpoint
    model = SmolLM2Lightning.load_from_checkpoint(
        checkpoint_path,
        vocab_size=VOCAB_SIZE,
        block_size=TRAIN_CONFIG["block_size"],
        lr=TRAIN_CONFIG["max_lr"],
        warmup_steps=TRAIN_CONFIG["warmup_steps"],
        max_steps=completed_steps + extra_steps,
        min_lr=TRAIN_CONFIG["min_lr"],
    )
    print("✅ Checkpoint loaded successfully!\n")
    
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
    
    trainer.fit(model, train_loader, val_loader)
    
    # Save final checkpoint
    final_ckpt = f"{CHECKPOINT_DIR}/smollm2-phase2-final.ckpt"
    trainer.save_checkpoint(final_ckpt)
    
    total_steps = completed_steps + extra_steps
    total_samples = calculate_sample_offset(total_steps)
    
    print(f"\n✅ Phase 2 complete!")
    print(f"Checkpoint saved to: {final_ckpt}")
    print(f"Total steps completed: {total_steps:,}")
    print(f"Total samples processed: {total_samples:,}\n")
    
    return final_ckpt


def main():
    seed_everything(SEED, workers=True)

    # Set precision
    torch.set_float32_matmul_precision('high')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    # Get dataset info
    dataset_info = get_dataset_info(tokenizer)
    
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Total tokens:        {dataset_info['total_tokens']:,}")
    print(f"Total samples:       {dataset_info['total_samples']:,}")
    print(f"Steps per epoch:     {dataset_info['steps_per_epoch']:,}")
    print(f"Block size:          {TRAIN_CONFIG['block_size']}")
    print(f"Batch size:          {TRAIN_CONFIG['batch_size']}")
    print("=" * 60 + "\n")
    
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
    # PHASE 1: Train from scratch (sample_offset = 0)
    # ============================================================
    train_loader, val_loader = create_dataloaders(tokenizer, sample_offset=0)
    
    model = SmolLM2Lightning(
        vocab_size=VOCAB_SIZE,
        block_size=TRAIN_CONFIG["block_size"],
        lr=TRAIN_CONFIG["max_lr"],
        warmup_steps=TRAIN_CONFIG["warmup_steps"],
        max_steps=TRAIN_CONFIG["max_steps"],
        min_lr=TRAIN_CONFIG["min_lr"],
        pretrained_path=None  # Train from scratch
    )
    
    phase1_ckpt, completed_steps = train_phase1(model, train_loader, val_loader, logger, dataset_info)
    
    # ============================================================
    # PHASE 2: Continue training with REMAINING data
    # ============================================================
    logger = TensorBoardLogger(LOG_DIR, name=EXPERIMENT_NAME+"_phase2")

    phase2_ckpt = train_phase2(
        phase1_ckpt, 
        tokenizer, 
        logger,
        dataset_info,
        completed_steps=completed_steps,
        extra_steps=50
    )
    
    print("\n" + "=" * 60)
    print("✅ ALL TRAINING PHASES COMPLETE!")
    print("=" * 60)
    print(f"Final checkpoint: {phase2_ckpt}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
