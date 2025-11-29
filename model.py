"""SmolLM2 Lightning Module."""

import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from config import MODEL_CONFIG, TOKENIZER_NAME


class SmolLM2Lightning(pl.LightningModule):
    """PyTorch Lightning module for SmolLM2 training."""
    
    def __init__(
        self, 
        vocab_size: int,
        block_size: int,
        lr: float,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        pretrained_path: str = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Build model config
        config_dict = MODEL_CONFIG.copy()
        config_dict["vocab_size"] = vocab_size
        config = LlamaConfig(**config_dict)
        
        # Initialize model
        self.model = LlamaForCausalLM(config)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load pretrained weights if specified
        if pretrained_path:
            self._load_pretrained(pretrained_path)
        
        # Log parameter count
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameter count: {param_count:,}")

    def _load_pretrained(self, pretrained_path: str):
        """Load pretrained weights."""
        pretrained_model = LlamaForCausalLM.from_pretrained(
            pretrained_path,
            torch_dtype=torch.bfloat16,
        )
        self.model.load_state_dict(pretrained_model.state_dict(), strict=True)
        print(f"✅ Pretrained weights loaded from: {pretrained_path}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
        """
        perplexity = torch.exp(loss)
        self.log('train_perplexity', perplexity, prog_bar=False, on_step=True, on_epoch=False, logger=True)
        """
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        """
        perplexity = torch.exp(loss)
        self.log('val_perplexity', perplexity, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        """
        # Generate sample text (first batch only)
        if batch_idx == 0:
            self._generate_sample(batch)

        return loss

    def _generate_sample(self, batch):
        """Generate sample text during validation."""
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

    def generate_text(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7):
        """Generate text from a prompt."""
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