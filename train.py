import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM #
from pytorch_lightning.loggers import TensorBoardLogger

# Dataset
class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, block_size=1024):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokenizer = tokenizer
        self.block_size = block_size
        # Encode without special tokens for raw text
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

        # Reverse engineer from SmolLM2 config
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

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Optional: Load pretrained weights for swap (set to None for random init validation)
        if pretrained_path:
            pretrained_model = LlamaForCausalLM.from_pretrained(
                pretrained_path,
                torch_dtype=torch.bfloat16,
            )
            self.model.load_state_dict(pretrained_model.state_dict(), strict=True)
            print("Pretrained weights loaded successfully.")

        # Verify param count (~135M)
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

        # Generate for validation (simple text gen to check coherence)
        with torch.no_grad():
            input_ids = batch["input_ids"]

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            gen_outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
            gen_text = self.tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
            print(f"Generated: {gen_text}")  # Log to console for validation

        return loss

    def configure_optimizers(self):
        # From YAML: AdamW with specific betas, eps, weight_decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01
        )

        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                # Linear warmup
                return float(current_step) / self.hparams.warmup_steps
            elif current_step >= self.hparams.max_steps:
                return self.hparams.min_lr / self.hparams.lr
            elif current_step < 1600000:
                # Constant LR after warmup
                return 1.0
            else:
                # Linear decay starting at 1600000 over 400000 steps to min_lr=0
                decay_start = 1600000
                decay_steps = 400000
                progress = (current_step - decay_start) / decay_steps
                return 1.0 - progress  # Decays to 0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# Training Script
if __name__ == "__main__":
    # Parameters (updated to match SmolLM2 YAML)
    block_size = 1024  # Can increase to 2048 if hardware allows
    batch_size = 8  # Smaller for 135M model
    max_lr = 0.003
    warmup_steps = 10
    max_steps = 5000  # Full pretrain steps; reduce for testing
    min_lr = 0.0

    # Tokenizer and vocab
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    vocab_size = 49152

    # Dataset and DataLoader (assumes input.txt exists; use a small file for testing)
    dataset = TextDataset("input.txt", tokenizer, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # For validation, create a small val dataset (e.g., dummy prompt)
    val_dataset = TextDataset("input.txt", tokenizer, block_size=128)  # Shorter for gen
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    torch.set_float32_matmul_precision('high')

    # Set up TensorBoard logger
    logger = TensorBoardLogger("logs/", name="smollm2_experiment")

    # Custom progress bar theme
    progress_bar = RichProgressBar(
        refresh_rate=1,
        leave=False,
        theme=RichProgressBarTheme(
            description='',
            progress_bar='#6206E0',
            progress_bar_finished='#6206E0',
            progress_bar_pulse='#6206E0',
            batch_progress='',
            time='dim',
            processing_speed='dim underline',
            metrics='italic',
            metrics_format='.3f'),
        console_kwargs=None
    )

    from pytorch_lightning.callbacks import TQDMProgressBar
    progress_bar = TQDMProgressBar(refresh_rate=1)

    """
    # Step 1: Init with random weights and validate (gibberish output)
    print("=== Random Weights Validation ===")
    random_model = SmolLM2Lightning(vocab_size, block_size, max_lr, warmup_steps, max_steps, pretrained_path=None)
    random_trainer = pl.Trainer(
        max_steps=0,  # No training, just validate
        accelerator="gpu",
        devices=1,
        callbacks=[LearningRateMonitor(logging_interval='step'), progress_bar],
        precision='bf16-mixed',
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=False  # Skip for random
    )
    random_trainer.validate(random_model, val_dataloader)
    """

    # Step 2: Swap to pretrained weights and validate (coherent output)
    print("\n=== Pretrained Weights Validation ===")
    pretrained_model = SmolLM2Lightning(
        vocab_size, 
        block_size, 
        max_lr, 
        warmup_steps, 
        max_steps, 
        min_lr=min_lr,  # Added this to match optimizer
        pretrained_path="HuggingFaceTB/SmolLM2-135M"
        )
    """
    pretrained_trainer = pl.Trainer(
        max_steps=0,
        accelerator="gpu",
        devices=1,
        callbacks=[LearningRateMonitor(logging_interval='step'), progress_bar],
        precision='bf16-mixed',
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=logger
    )
    #pretrained_trainer.validate(pretrained_model, val_dataloader)
    """
    # Optional: Proceed to training with pretrained (uncomment below)
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator="gpu",
        devices=1,
        callbacks=[LearningRateMonitor(logging_interval='step'), progress_bar],
        precision='bf16-mixed',
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=logger
    )
    trainer.fit(pretrained_model, dataloader, val_dataloader)


