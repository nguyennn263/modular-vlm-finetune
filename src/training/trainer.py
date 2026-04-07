"""
Clean training pipeline for Vision-Language fine-tuning.

Only trains bridge modules that convert:
  Vision embeddings (4096 dims) → LLM embeddings (896 dims)

Vision Model and Language Model are completely frozen.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Tuple, Dict, Optional
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoTokenizer

from src.middleware.logger import data_loader_logger as logger
from src.schema.data_schema import OneSample
from src.data.collator_onesample import create_collate_fn
from transformers import AutoTokenizer


@dataclass
class TrainConfig:
    """Training configuration."""
    model_name: str = "5CD-AI/Vintern-1B-v3_5"  # Model identifier for tokenizer
    output_dir: str = "checkpoints/bridge_experiments"
    num_epochs: int = 10
    batch_size: int = 2  # Reduced from 8 to fit in 14GB GPU memory
    gradient_accumulation_steps: int = 4  # Accumulate 4 batches for effective batch size of 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    eval_steps: int = 100
    save_steps: int = 500
    seed: int = 42
    device: str = "auto"
    fp16: bool = False
    num_workers: int = 4
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.001
    
    # Checkpoint
    resume_from: Optional[str] = None
    save_best: bool = True


class BridgeTrainer:
    """
    Trainer for bridge modules in vision-language models.
    
    Key features:
    - Vision and Language models are frozen
    - Only bridge module is trained
    - Automatic device selection
    - Checkpoint save/resume
    - Early stopping with patience
    """
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config: TrainConfig,
        test_dataset=None
    ):
        self.config = config
        self.device = self._get_device()
        
        # Optimize CUDA memory allocations
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Model and data
        self.model = model.to(self.device)
        
        # Ensure bridge module is in same dtype as vision model (bfloat16)
        model_dtype = next(self.model.vision_model.parameters()).dtype
        if hasattr(self.model, 'bridge'):
            self.model.bridge = self.model.bridge.to(dtype=model_dtype)
        
        # Disable gradient checkpointing on frozen models to eliminate warnings
        if hasattr(self.model.vision_model, 'gradient_checkpointing_disable'):
            self.model.vision_model.gradient_checkpointing_disable()
        if hasattr(self.model.language_model, 'gradient_checkpointing_disable'):
            self.model.language_model.gradient_checkpointing_disable()
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Data loaders
        # Use custom collate function if datasets contain OneSample objects
        collate_fn = None
        tokenizer = None
        
        if train_dataset and len(train_dataset) > 0:
            if isinstance(train_dataset[0], OneSample):
                # Load tokenizer from model_name (like in the notebook)
                try:
                    logger.info(f"Loading tokenizer from: {self.config.model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.config.model_name,
                        trust_remote_code=True,
                        use_fast=False
                    )
                    logger.info("✓ Tokenizer loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load tokenizer: {e}")
                    raise
                
                # Use max_length=256 for memory efficiency (reduces memory usage by ~50%)
                # This is still plenty for Q&A pairs
                collate_fn = create_collate_fn(
                    tokenizer=tokenizer,
                    image_size=(336, 336),
                    max_length=256  # Reduced from default 512
                )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            collate_fn=collate_fn
        )
        
        # Optional test loader
        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=(self.device.type == 'cuda'),
                collate_fn=collate_fn
            )
        
        # Setup optimization
        self._setup_optimization()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.best_model_path = None
        
        # Resume if specified
        if config.resume_from and os.path.exists(config.resume_from):
            self._load_checkpoint(config.resume_from)
        
        self._log_info()
    
    def _get_device(self) -> torch.device:
        """Get training device with fallback."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                logger.info("✓ Using CUDA")
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("⚠ Using Metal Performance Shaders (MPS)")
                return torch.device("mps")
            else:
                logger.warning("⚠ GPU not found, using CPU (training will be slow)")
                return torch.device("cpu")
        elif self.config.device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                logger.warning("⚠ CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def _setup_optimization(self):
        """Setup optimizer and scheduler."""
        # Freeze base models
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        
        for param in self.model.language_model.parameters():
            param.requires_grad = False
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_count:,} ({100*trainable_count/total_params:.2f}%)")
        logger.info(f"Frozen parameters: {total_params - trainable_count:,}")
        
        # Optimizer
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
    
    def _log_info(self):
        """Log training configuration."""
        logger.info("=" * 80)
        logger.info("Bridge Module Fine-tuning")
        logger.info("=" * 80)
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Early stopping: {'Enabled' if self.config.early_stopping else 'Disabled'}")
        if self.config.early_stopping:
            logger.info(f"  Patience: {self.config.patience}")
            logger.info(f"  Min delta: {self.config.min_delta}")
        logger.info("=" * 80)
    
    def forward_pass(self, batch: Dict) -> torch.Tensor:
        """
        Forward pass: vision → bridge → combined embeddings → LLM.
        
        Args:
            batch: Dict with pixel_values, input_ids, attention_mask
        
        Returns:
            Loss tensor
        """
        # Get model dtype (check vision model dtype)
        model_dtype = next(self.model.vision_model.parameters()).dtype
        
        # Convert pixel_values to model dtype and device (matching notebook pattern)
        pixel_values = batch['pixel_values'].to(dtype=model_dtype, device=self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Get vision embeddings (frozen model, but no_grad not needed for bridge input)
        vision_output = self.model.vision_model(pixel_values)
        # Extract tensor from BaseModelOutputWithPooling
        # Different bridges need different input shapes:
        # - BetterMLP: pooled vector [batch, 1024]
        # - Others (MultiTokenMLP, AttentionBridge, etc): full sequence [batch, num_patches, 1024]
        bridge_type = getattr(self.model, 'bridge_type', 'unknown')
        
        if bridge_type == 'better_mlp':
            # BetterMLP expects pooled single vector
            if hasattr(vision_output, 'pooler_output'):
                vision_embeddings = vision_output.pooler_output
            elif hasattr(vision_output, 'last_hidden_state'):
                vision_embeddings = vision_output.last_hidden_state[:, 0, :]  # CLS token
            else:
                vision_embeddings = vision_output
        else:
            # MultiToken, Attention, MiniQFormer, QFormer need full patch sequence
            if hasattr(vision_output, 'last_hidden_state'):
                vision_embeddings = vision_output.last_hidden_state  # [batch, num_patches, 1024]
            elif hasattr(vision_output, 'pooler_output'):
                # Fallback: unsqueeze pooler output to create sequence format
                vision_embeddings = vision_output.pooler_output.unsqueeze(1)
            else:
                if vision_output.dim() == 2:
                    vision_embeddings = vision_output.unsqueeze(1)
                else:
                    vision_embeddings = vision_output
        
        # Detach vision embeddings since vision model is frozen
        # This prevents any gradient computation in the vision model
        vision_embeddings = vision_embeddings.detach()
        
        # Apply bridge module (trainable)
        # Bridge handles both shape conversion and augmentation
        # Input: [batch_size, vision_dim] -> Output: [batch_size, text_dim]
        bridged_embeddings = self.model.bridge(vision_embeddings)
        
        # Get text embeddings (frozen)
        text_embeddings = self.model.language_model.model.embed_tokens(input_ids)
        # Convert to model dtype (embeddings are float32 by default)
        text_embeddings = text_embeddings.to(dtype=model_dtype)
        
        # Detach text embeddings since language model is frozen
        # This prevents any gradient computation in the language model embeddings
        text_embeddings = text_embeddings.detach()
        
        # Ensure bridged_embeddings has sequence dimension [batch_size, 1, text_dim]
        # So it can be concatenated with text_embeddings [batch_size, seq_len, text_dim]
        # Also convert to model dtype
        if bridged_embeddings.dim() == 2:
            bridged_embeddings = bridged_embeddings.unsqueeze(1).to(dtype=model_dtype)
        else:
            bridged_embeddings = bridged_embeddings.to(dtype=model_dtype)
        
        # Combine vision and text embeddings
        combined_embeddings = torch.cat([bridged_embeddings, text_embeddings], dim=1)
        
        # Create attention mask for combined embeddings
        # Vision contribution is num_vision_tokens, text is rest
        num_vision_tokens = bridged_embeddings.shape[1]
        vision_attention = torch.ones(
            bridged_embeddings.shape[0],
            num_vision_tokens,
            device=self.device,
            dtype=attention_mask.dtype
        )
        combined_attention_mask = torch.cat([vision_attention, attention_mask], dim=1)
        
        # Forward through LLM (frozen model, but compute logits to backprop through bridge)
        outputs = self.model.language_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits
        
        # Compute loss (next-token prediction) on text tokens only
        # logits has shape [batch_size, num_vision_tokens + text_len, vocab_size]
        # Skip all vision tokens and use only text logits
        text_logits = logits[:, num_vision_tokens:, :]  # [batch_size, text_len, vocab_size]
        
        # Shift for next-token prediction
        shift_logits = text_logits[..., :-1, :].contiguous()  # [batch_size, text_len-1, vocab_size]
        shift_labels = input_ids[..., 1:].contiguous()  # [batch_size, text_len-1]
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction='mean'
        )
        
        return loss
    
    def train_epoch(self, epoch: int) -> Tuple[float, bool]:
        """
        Train for one epoch with gradient accumulation.
        
        Returns:
            (avg_loss, should_stop)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulation_counter = 0
        accumulated_loss = 0.0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            loss = self.forward_pass(batch)
            
            # Scale loss by accumulation steps (gradient accumulation divides loss)
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass (accumulate gradients)
            loss.backward()
            accumulated_loss += loss.item()
            accumulation_counter += 1
            
            # Update weights every N accumulation steps
            if accumulation_counter % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.bridge.parameters() if p.requires_grad],
                    max_norm=self.config.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Logging
                total_loss += accumulated_loss
                num_batches += 1
                self.global_step += 1
                
                pbar.set_postfix({'loss': f'{accumulated_loss:.4f}'})
                
                # Validation and checkpoint
                if self.global_step % self.config.eval_steps == 0:
                    val_loss = self.validate()
                    logger.info(f"Step {self.global_step}: train_loss={accumulated_loss:.4f}, val_loss={val_loss:.4f}")
                    
                    # Check for improvement
                    is_best = val_loss < self.best_val_loss - self.config.min_delta
                    
                    if is_best:
                        self.best_val_loss = val_loss
                        self.early_stop_counter = 0
                        self.save_checkpoint(is_best=True)
                        logger.info(f"✓ New best: val_loss={val_loss:.4f}")
                    else:
                        self.early_stop_counter += 1
                        if self.config.early_stopping and self.early_stop_counter >= self.config.patience:
                            logger.warning(f"⚠ Early stopping at step {self.global_step}")
                            return total_loss / num_batches if num_batches > 0 else 0.0, True
                    
                    # Clear cache after validation to free memory for training
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                
                # Regular checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Reset accumulation
                accumulated_loss = 0.0
        
        # Handle remaining accumulated gradients
        if accumulation_counter % self.config.gradient_accumulation_steps != 0:
            nn.utils.clip_grad_norm_(
                [p for p in self.model.bridge.parameters() if p.requires_grad],
                max_norm=self.config.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            total_loss += accumulated_loss
            num_batches += 1
            self.global_step += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0, False
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate on validation set and compute loss metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        for batch in pbar:
            loss = self.forward_pass(batch)
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    @torch.no_grad()
    def evaluate(self, test_dataset=None) -> Dict[str, float]:
        """
        Evaluate on test set with comprehensive metrics.
        
        Args:
            test_dataset: Optional test dataset. If None, uses val_dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        from dataclasses import dataclass
        
        self.model.eval()
        
        # Use test dataset if provided, otherwise use validation
        eval_loader = self.test_loader if hasattr(self, 'test_loader') and self.test_loader else self.val_loader
        
        total_loss = 0.0
        num_batches = 0
        all_metrics = {
            'loss': 0.0,
            'num_samples': 0,
        }
        
        pbar = tqdm(eval_loader, desc="Evaluating", leave=False)
        
        for batch in pbar:
            loss = self.forward_pass(batch)
            total_loss += loss.item()
            num_batches += 1
            all_metrics['num_samples'] += batch['input_ids'].shape[0]
            
            pbar.set_postfix({'eval_loss': f'{loss.item():.4f}'})
        
        all_metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        all_metrics['perplexity'] = torch.exp(torch.tensor(all_metrics['loss'])).item()
        
        self.model.train()
        return all_metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'bridge_state': self.model.bridge.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'early_stop_counter': self.early_stop_counter,
        }
        
        if is_best:
            path = os.path.join(self.config.output_dir, 'best_model.pt')
            self.best_model_path = path
        else:
            path = os.path.join(self.config.output_dir, f'step_{self.global_step}.pt')
        
        torch.save(checkpoint, path)
        logger.info(f"✓ Saved {'best ' if is_best else ''}checkpoint: {path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.bridge.load_state_dict(checkpoint['bridge_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.early_stop_counter = checkpoint['early_stop_counter']
        
        logger.info(f"✓ Resumed from checkpoint (step {self.global_step})")
    
    def train(self):
        """Main training loop."""
        start_time = datetime.now()
        
        try:
            for epoch in range(self.config.num_epochs):
                avg_loss, should_stop = self.train_epoch(epoch)
                logger.info(f"Epoch {epoch+1} completed: avg_loss={avg_loss:.4f}")
                
                if should_stop:
                    break
            
            # Save final model
            if self.config.save_best and self.best_model_path:
                logger.info(f"✓ Best model at: {self.best_model_path}")
        
        except Exception as e:
            logger.error(f"✗ Training failed: {e}")
            raise
        
        finally:
            elapsed = datetime.now() - start_time
            logger.info(f"Training completed in {elapsed}")


# Backward compatibility
BridgeFineTuner = BridgeTrainer
