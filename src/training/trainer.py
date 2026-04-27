"""
Clean training pipeline for Vision-Language fine-tuning.

Only trains bridge modules that convert:
  Vision embeddings (4096 dims) → LLM embeddings (896 dims)

Vision Model and Language Model are completely frozen.
"""

import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Tuple, Dict, Optional, List
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoTokenizer
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from src.middleware.logger import data_loader_logger as logger
from src.schema.data_schema import OneSample
from src.data.collator_onesample import create_collate_fn


# ============================================================================
# IMAGE PREPROCESSING FROM NOTEBOOK
# ============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    """Build image transformation pipeline with ImageNet normalization."""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio for image tiling."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12, 
                      image_size: int = 448, use_thumbnail: bool = False) -> List[Image.Image]:
    """Dynamically preprocess image by dividing into patches."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    # Extract patches
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    # Add thumbnail if needed
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    """Load and preprocess image for model input."""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
# ============================================================================


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
        
        # Setup result tracking (per-epoch and logs)
        self._setup_result_tracking()
        
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
                
                # Store tokenizer for inference
                self.tokenizer = tokenizer
                
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
        
        # Checkpoint management: track recent checkpoints for cleanup
        self.recent_checkpoints = []  # Keep only 2 most recent
        
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
        
        # Scheduler with warmup
        # Scheduler should follow optimizer steps, not micro-batches.
        steps_per_epoch = math.ceil(len(self.train_loader) / self.config.gradient_accumulation_steps)
        total_steps = max(steps_per_epoch * self.config.num_epochs, 1)
        warmup_steps = min(self.config.warmup_steps, total_steps // 10)  # Cap warmup at 10% of total
        
        # Use CosineAnnealingLR with warmup
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step):
            """Linear warmup followed by cosine decay."""
            if current_step < warmup_steps:
                # Linear warmup: 0 to 1 over warmup_steps
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine annealing: 1 to eta_min over remaining steps
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(1e-6 / self.config.learning_rate, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        logger.info(f"Warmup steps: {warmup_steps}/{total_steps}")
        logger.info(f"Total training steps: {total_steps}")
    
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
    
    def _setup_result_tracking(self):
        """Setup per-epoch result tracking and log files."""
        import csv
        
        # Create results directory
        results_dir = Path(self.config.output_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-epoch results CSV
        self.epoch_results_file = results_dir / "epoch_results.csv"
        self.epoch_results_writer = None
        self.epoch_results_csv = None
        
        # Training logs file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = results_dir / f"training_{timestamp}.log"
        
        # Write header to CSV
        try:
            with open(self.epoch_results_file, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'epoch', 'global_step',
                        'train_loss', 'val_loss',
                        'perplexity', 'learning_rate',
                        'metric_bleu', 'metric_meteor', 'metric_rouge_l',
                        'metric_cider', 'metric_exact_match',
                        'metric_precision', 'metric_recall', 'metric_f1',
                        'metric_wups',
                        'early_stop_counter', 'is_best',
                        'time_seconds'
                    ]
                )
                writer.writeheader()
        except Exception as e:
            logger.warning(f"Failed to create epoch results CSV: {e}")
        
        logger.info(f"✓ Results will be saved to: {results_dir}")
        logger.info(f"✓ Epoch results: {self.epoch_results_file}")
        logger.info(f"✓ Training logs: {self.log_file}")
    
    def _log_to_file(self, message: str):
        """Write message to both logger and log file."""
        logger.info(message)
        try:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
        except Exception:
            pass  # Silently fail if file write fails
    
    def _save_epoch_results(self, epoch: int, epoch_metrics: Dict):
        """Save per-epoch results to CSV."""
        import csv
        
        try:
            with open(self.epoch_results_file, 'a', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'epoch', 'global_step',
                        'train_loss', 'val_loss',
                        'perplexity', 'learning_rate',
                        'metric_bleu', 'metric_meteor', 'metric_rouge_l',
                        'metric_cider', 'metric_exact_match',
                        'metric_precision', 'metric_recall', 'metric_f1',
                        'metric_wups',
                        'early_stop_counter', 'is_best',
                        'time_seconds'
                    ]
                )
                writer.writerow({
                    'epoch': epoch + 1,
                    'global_step': self.global_step,
                    'train_loss': epoch_metrics.get('train_loss', 0),
                    'val_loss': epoch_metrics.get('val_loss', 0),
                    'perplexity': epoch_metrics.get('perplexity', 0),
                    'learning_rate': epoch_metrics.get('learning_rate', 0),
                    'metric_bleu': epoch_metrics.get('metric_bleu', 0),
                    'metric_meteor': epoch_metrics.get('metric_meteor', 0),
                    'metric_rouge_l': epoch_metrics.get('metric_rouge_l', 0),
                    'metric_cider': epoch_metrics.get('metric_cider', 0),
                    'metric_exact_match': epoch_metrics.get('metric_exact_match', 0),
                    'metric_precision': epoch_metrics.get('metric_precision', 0),
                    'metric_recall': epoch_metrics.get('metric_recall', 0),
                    'metric_f1': epoch_metrics.get('metric_f1', 0),
                    'metric_wups': epoch_metrics.get('metric_wups', 0),
                    'early_stop_counter': self.early_stop_counter,
                    'is_best': epoch_metrics.get('is_best', False),
                    'time_seconds': epoch_metrics.get('time_seconds', 0)
                })
        except Exception as e:
            logger.warning(f"Failed to save epoch results: {e}")
    
    def _save_final_summary(self, elapsed_timedelta, total_hours: float):
        """Save final training summary to JSON with all metrics and results."""
        import json
        
        try:
            results_dir = Path(self.config.output_dir) / "results"
            summary_file = results_dir / "summary.json"
            metrics_summary_file = results_dir / "final_metrics_summary.json"
            
            summary = {
                'experiment': Path(self.config.output_dir).name,
                'best_val_loss': float(self.best_val_loss),
                'best_model_path': str(self.best_model_path) if self.best_model_path else None,
                'global_step': self.global_step,
                'training_duration': str(elapsed_timedelta),
                'training_duration_hours': total_hours,
                'epochs_trained': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'early_stopping_enabled': self.config.early_stopping,
                'results_files': {
                    'epoch_results': str(self.epoch_results_file),
                    'training_logs': str(self.log_file),
                    'summary': str(summary_file),
                    'text_metrics_all_epochs': str(results_dir / "text_metrics_all_epochs.jsonl"),
                    'metrics_summary': str(metrics_summary_file)
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Also create a metrics-only summary for quick reference
            metrics_summary = {
                'description': 'Final training metrics summary (ref/ref1-aligned)',
                'metrics_format': {
                    'bleu': 'BLEU-4 score',
                    'meteor': 'METEOR score',
                    'rouge_l': 'ROUGE-L score',
                    'cider': 'CIDEr score',
                    'exact_match': 'Exact match accuracy',
                    'precision': 'Word-level precision',
                    'recall': 'Word-level recall',
                    'f1': 'Word-level F1 score',
                    'wups': 'WUPS@0.9 score'
                },
                'output_files': {
                    'epoch_results_csv': str(self.epoch_results_file),
                    'per_epoch_metrics': str(results_dir / "text_metrics_epoch_*.json"),
                    'per_sample_predictions': str(results_dir / "text_predictions_epoch_*.json"),
                    'cumulative_metrics': str(results_dir / "text_metrics_all_epochs.jsonl"),
                    'training_log': str(self.log_file)
                }
            }
            
            with open(metrics_summary_file, 'w') as f:
                json.dump(metrics_summary, f, indent=2)
            
            logger.info(f"✓ Final summary saved to: {summary_file}")
            logger.info(f"✓ Metrics summary saved to: {metrics_summary_file}")
            logger.info(f"✓ Training logs saved to: {self.log_file}")
            logger.info(f"✓ All results saved to: {results_dir}")
        except Exception as e:
            logger.warning(f"Failed to save final summary: {e}")
    
    
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
        
        # Extract vision features - debug what we have
        if hasattr(vision_output, 'last_hidden_state'):
            last_hidden = vision_output.last_hidden_state  # Often [batch, num_patches, dim]
            pooler = vision_output.pooler_output if hasattr(vision_output, 'pooler_output') else None
        elif hasattr(vision_output, 'pooler_output'):
            last_hidden = None
            pooler = vision_output.pooler_output
        else:
            last_hidden = vision_output if isinstance(vision_output, torch.Tensor) else None
            pooler = None
        
        # Decide which to use based on bridge type
        if bridge_type in ['linear_bridge', 'better_mlp', 'multi_token']:
            # LinearBridge, BetterMLP + MultiTokenMLP expect single pooled vector [batch, 1024]
            # Both expand to multiple tokens internally
            if pooler is not None:
                vision_embeddings = pooler
            elif last_hidden is not None:
                vision_embeddings = last_hidden[:, 0, :]  # Use CLS token
            else:
                vision_embeddings = vision_output
        else:
            # AttentionBridge, MiniQFormer, QFormer need full sequence [batch, num_patches, 1024]
            if last_hidden is not None and last_hidden.dim() == 3:
                vision_embeddings = last_hidden  # Full sequence
            elif last_hidden is not None and last_hidden.dim() == 2:
                # If last_hidden is 2D, unsqueeze it
                vision_embeddings = last_hidden.unsqueeze(1)
            elif pooler is not None:
                # Use pooler and unsqueeze to create sequence
                vision_embeddings = pooler.unsqueeze(1)
            else:
                # Fallback
                if isinstance(vision_output, torch.Tensor):
                    if vision_output.dim() == 2:
                        vision_embeddings = vision_output.unsqueeze(1)
                    else:
                        vision_embeddings = vision_output
                else:
                    raise ValueError(f"Cannot extract vision embeddings from {type(vision_output)}")
        
        # Detach vision embeddings since vision model is frozen
        # This prevents any gradient computation in the vision model
        vision_embeddings = vision_embeddings.detach()
        
        # Validate shapes before passing to bridge
        if bridge_type in ['linear_bridge', 'better_mlp', 'multi_token']:
            # These expect 2D pooled vectors [batch, 1024]
            assert vision_embeddings.dim() == 2, (
                f"Bridge {bridge_type} expects 2D vision_embeddings [batch, dim], "
                f"got shape {vision_embeddings.shape} (dim={vision_embeddings.dim()})"
            )
        else:
            # Others expect 3D sequences [batch, seq, dim]
            assert vision_embeddings.dim() == 3, (
                f"Bridge {bridge_type} expects 3D vision_embeddings [batch, seq, dim], "
                f"got shape {vision_embeddings.shape} (dim={vision_embeddings.dim()})"
            )
        
        # Get text embeddings early (needed for QFormer and concatenation)
        text_embeddings = self.model.language_model.model.embed_tokens(input_ids)
        # Convert to model dtype immediately (embeddings are float32 by default)
        text_embeddings = text_embeddings.to(dtype=model_dtype, device=self.device)
        
        # IMPORTANT: Only detach for non-QFormer bridges!
        # QFormer needs gradients through text embeddings for semantic understanding
        if bridge_type != 'qformer':
            # Detach text embeddings since language model is frozen
            # This prevents any gradient computation in the language model embeddings
            text_embeddings = text_embeddings.detach()
        # For QFormer: keep text embeddings on computation graph so bridge learns semantics
        
        # Apply bridge module (trainable)
        # Bridge handles both shape conversion and augmentation
        if bridge_type == 'qformer':
            # QFormer requires both vision features and question embeddings
            bridged_embeddings = self.model.bridge(vision_embeddings, text_embeddings)
        else:
            # All other bridges just take vision embeddings
            bridged_embeddings = self.model.bridge(vision_embeddings)
        
        # Ensure bridged_embeddings has sequence dimension [batch_size, num_tokens, text_dim]
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

        # Compute loss only on valid answer tokens (exclude padding and prompt tokens).
        loss_mask = attention_mask[..., 1:].contiguous().bool()
        if 'answer_start_pos' in batch:
            answer_start_pos = batch['answer_start_pos'].to(self.device)
            token_positions = torch.arange(1, input_ids.shape[1], device=self.device).unsqueeze(0)
            answer_mask = token_positions >= answer_start_pos.unsqueeze(1)
            loss_mask = loss_mask & answer_mask

        if not loss_mask.any():
            return shift_logits.sum() * 0.0

        masked_labels = shift_labels.masked_fill(~loss_mask, -100)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            masked_labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        # Add distillation loss to prevent embedding collapse
        # This keeps embeddings on the valid LLM manifold during training
        use_distillation = True  # Can be made configurable
        distillation_weight = 0.5  # Default weight for MSE loss
        
        if use_distillation:
            # Get baseline embeddings (frozen reference)
            with torch.no_grad():
                # Baseline always uses pooled vision features
                if vision_embeddings.dim() == 3:
                    vision_pool_baseline = vision_embeddings[:, 0, :]  # CLS token
                else:
                    vision_pool_baseline = vision_embeddings
                baseline_output = self.model.baseline_bridge(vision_pool_baseline)  # (B, 896)
                baseline_output = baseline_output.to(dtype=model_dtype, device=self.device)
            
            # For distillation, compare first token of bridge output (which is the main output)
            if bridged_embeddings.dim() == 3:
                bridge_first_token = bridged_embeddings[:, 0, :]  # (B, 896)
            else:
                bridge_first_token = bridged_embeddings
            
            # MSE loss to keep bridge output close to baseline
            distillation_loss = F.mse_loss(bridge_first_token, baseline_output)
            
            # Total loss = CE + lambda * MSE
            loss = loss + distillation_weight * distillation_loss
        
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

        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            loss = self.forward_pass(batch)
            raw_loss = loss

            # Scale loss by accumulation steps (gradient accumulation divides loss)
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass (accumulate gradients)
            loss.backward()
            accumulated_loss += raw_loss.item()
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
                self.optimizer.zero_grad(set_to_none=True)

                step_loss = accumulated_loss / self.config.gradient_accumulation_steps
                
                # Logging
                total_loss += step_loss
                num_batches += 1
                self.global_step += 1
                
                pbar.set_postfix({'loss': f'{step_loss:.4f}'})
                
                # Validation and checkpoint
                if self.global_step % self.config.eval_steps == 0:
                    val_loss = self.validate()
                    logger.info(f"Step {self.global_step}: train_loss={step_loss:.4f}, [VAL] val_loss={val_loss:.4f}")
                    
                    # Show sample inference results every validation
                    import random
                    if hasattr(self, 'val_dataset') and len(self.val_dataset) > 0:
                        try:
                            indices = random.sample(range(len(self.val_dataset)), min(3, len(self.val_dataset)))
                            logger.info(f"\n{'='*80}")
                            logger.info(f"Sample Inference - Step {self.global_step}")
                            logger.info(f"{'='*80}")
                            
                            self.model.eval()
                            for i, idx in enumerate(indices, 1):
                                sample = self.val_dataset[idx]
                                question = sample.question if hasattr(sample, 'question') else 'N/A'
                                # Handle answers list (new schema) - use majority vote like collator does
                                if hasattr(sample, 'answers'):
                                    sample_answers = sample.answers if isinstance(sample.answers, list) else [sample.answers]
                                    if len(sample_answers) == 0:
                                        answer = 'N/A'
                                    elif len(sample_answers) == 1:
                                        answer = sample_answers[0]
                                    else:
                                        # Majority vote: get most common answer (same as collator)
                                        from collections import Counter
                                        counter = Counter(sample_answers)
                                        answer = counter.most_common(1)[0][0]
                                else:
                                    answer = 'N/A'
                                
                                # Try to get model prediction using notebook approach
                                try:
                                    # Load image using dynamic preprocessing from notebook
                                    pixel_values = load_image(
                                        sample.image_path, 
                                        input_size=448, 
                                        max_num=6
                                    ).to(self.device)
                                    
                                    # Get model dtype from vision model
                                    model_dtype = next(self.model.vision_model.parameters()).dtype
                                    pixel_values = pixel_values.to(dtype=model_dtype)
                                    
                                    # Inference with no grad
                                    with torch.no_grad():
                                        # Get vision embeddings via bridge
                                        if pixel_values.dim() == 4:
                                            pixel_values_input = pixel_values[0:1, :, :, :]  # [1, 3, 448, 448]
                                        else:
                                            pixel_values_input = pixel_values.unsqueeze(0)
                                        vision_output = self.model.vision_model(pixel_values_input)
                                        
                                        # Extract tensor from BaseModelOutputWithPooling
                                        bridge_type = getattr(self.model, 'bridge_type', 'unknown')
                                        if hasattr(vision_output, 'last_hidden_state'):
                                            last_hidden = vision_output.last_hidden_state
                                            pooler = vision_output.pooler_output if hasattr(vision_output, 'pooler_output') else None
                                        elif hasattr(vision_output, 'pooler_output'):
                                            last_hidden = None
                                            pooler = vision_output.pooler_output
                                        else:
                                            last_hidden = vision_output if isinstance(vision_output, torch.Tensor) else None
                                            pooler = None
                                        
                                        # Decide which to use based on bridge type
                                        if bridge_type in ['linear_bridge', 'better_mlp', 'multi_token']:
                                            if pooler is not None:
                                                vision_embeddings = pooler
                                            elif last_hidden is not None:
                                                vision_embeddings = last_hidden[:, 0, :]
                                            else:
                                                raise ValueError(f"Cannot extract vision embeddings for {bridge_type}")
                                        else:
                                            if last_hidden is not None and last_hidden.dim() == 3:
                                                vision_embeddings = last_hidden
                                            elif last_hidden is not None and last_hidden.dim() == 2:
                                                vision_embeddings = last_hidden.unsqueeze(1)
                                            elif pooler is not None:
                                                vision_embeddings = pooler.unsqueeze(1)
                                            else:
                                                raise ValueError(f"Cannot extract vision embeddings for {bridge_type}")
                                        
                                        vision_embeddings = vision_embeddings.detach()
                                    
                                    # Prepare prompt matching training format
                                    system_message = "Bạn là một mô hình trí tuệ nhân tạo đa phương thức Tiếng Việt có tên gọi là Vintern, được phát triển bởi người Việt. Bạn là một trợ lý trí tuệ nhân tạo hữu ích và không gây hại."
                                    prompt_text = (
                                        f"<|im_start|>system\n{system_message}<|im_end|>\n"
                                        f"<|im_start|>user\n<image>\n{question}<|im_end|>\n"
                                        f"<|im_start|>assistant\n"
                                    )
                                    
                                    with torch.no_grad():
                                        # Tokenize prompt
                                        inputs = self.tokenizer(
                                            prompt_text,
                                            return_tensors='pt',
                                            padding=False,
                                            truncation=True,
                                            max_length=512
                                        )
                                        input_ids = inputs['input_ids'].to(self.device)
                                        attention_mask = inputs['attention_mask'].to(self.device)
                                        
                                        # Get text embeddings
                                        text_embeddings = self.model.language_model.model.embed_tokens(input_ids)
                                        # Convert to model dtype immediately
                                        text_embeddings = text_embeddings.to(dtype=model_dtype, device=self.device)

                                        # QFormer needs question/text context for bridge generation.
                                        if bridge_type == 'qformer':
                                            bridged_embeddings = self.model.bridge(vision_embeddings, text_embeddings)
                                        else:
                                            bridged_embeddings = self.model.bridge(vision_embeddings)
                                        if bridged_embeddings.dim() == 2:
                                            bridged_embeddings = bridged_embeddings.unsqueeze(1)
                                        
                                        # Combine vision + text embeddings
                                        combined = torch.cat([bridged_embeddings, text_embeddings], dim=1)
                                        vision_attention = torch.ones(
                                            bridged_embeddings.shape[0], 
                                            bridged_embeddings.shape[1], 
                                            device=self.device, 
                                            dtype=attention_mask.dtype
                                        )
                                        combined_attention = torch.cat([vision_attention, attention_mask], dim=1)
                                        
                                        # Use generate() to get output
                                        outputs = self.model.language_model.generate(
                                            inputs_embeds=combined,
                                            attention_mask=combined_attention,
                                            max_new_tokens=50,
                                            do_sample=False,
                                            num_beams=1,
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            eos_token_id=self.tokenizer.eos_token_id,
                                            temperature=1.0,
                                            top_p=1.0
                                        )
                                        
                                        # Decode answer robustly for inputs_embeds generation.
                                        generated_ids = outputs[0]
                                        prompt_len = input_ids.shape[1]
                                        if generated_ids.shape[0] > prompt_len:
                                            output_ids = generated_ids[prompt_len:]
                                        else:
                                            output_ids = generated_ids
                                        model_output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                                        if not model_output:
                                            model_output = "[Empty output]"
                                    
                                except Exception as e:
                                    model_output = f"[Generation failed: {str(e)[:80]}]"
                                
                                logger.info(f"\n[Sample {i}]")
                                logger.info(f"Input: {question}")
                                logger.info(f"Model Output: {model_output}")
                                logger.info(f"Expected Output: {answer}")
                            
                            self.model.train()
                        except Exception as e:
                            logger.warning(f"Sample inference display failed: {e}")
                    
                    # Check for improvement
                    is_best = val_loss < self.best_val_loss - self.config.min_delta
                    
                    if is_best:
                        self.best_val_loss = val_loss
                        self.early_stop_counter = 0
                        self.save_checkpoint(is_best=True)
                        logger.info(f"✓ New best [VAL]: val_loss={val_loss:.4f}")
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
            self.optimizer.zero_grad(set_to_none=True)
            remaining_steps = accumulation_counter % self.config.gradient_accumulation_steps
            step_loss = accumulated_loss / remaining_steps
            total_loss += step_loss
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
        
        # Cleanup old checkpoints if not the best one
        if not is_best:
            self._cleanup_old_checkpoints(path)
    
    def _cleanup_old_checkpoints(self, new_checkpoint_path: str, keep_last_n: int = 1):
        """Keep only the N most recent checkpoints (besides best_model.pt)."""
        try:
            # Add the new checkpoint to the list
            self.recent_checkpoints.append(new_checkpoint_path)
            
            # Keep only the last N checkpoints
            if len(self.recent_checkpoints) > keep_last_n:
                old_checkpoint = self.recent_checkpoints.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    logger.info(f"✓ Removed old checkpoint: {old_checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")
    
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

    def _extract_sample_answers(self, sample) -> List[str]:
        """Extract all valid ground-truth answers from a dataset sample."""
        if not hasattr(sample, 'answers'):
            return ['']
        answers = sample.answers if isinstance(sample.answers, list) else [sample.answers]
        answers = [str(a).strip() for a in answers if a is not None and str(a).strip()]
        return answers if answers else ['']

    def _build_prompt_text(self, question: str) -> str:
        """Build prompt text in the same format as training/inference."""
        system_message = (
            "Bạn là một mô hình trí tuệ nhân tạo đa phương thức Tiếng Việt có tên gọi là Vintern, "
            "được phát triển bởi người Việt. Bạn là một trợ lý trí tuệ nhân tạo hữu ích và không gây hại."
        )
        return (
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
            f"<|im_start|>user\n<image>\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @torch.no_grad()
    def _generate_answer_for_sample(self, sample) -> str:
        """Generate one answer string for a validation sample."""
        question = sample.question if hasattr(sample, 'question') else 'N/A'
        pixel_values = load_image(sample.image_path, input_size=448, max_num=6)
        model_dtype = next(self.model.vision_model.parameters()).dtype
        pixel_values = pixel_values.to(dtype=model_dtype, device=self.device)

        prompt_text = self._build_prompt_text(question)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        if pixel_values.dim() == 4:
            pixel_values_input = pixel_values[0:1, :, :, :]
        else:
            pixel_values_input = pixel_values.unsqueeze(0)
        vision_output = self.model.vision_model(pixel_values_input)

        bridge_type = getattr(self.model, 'bridge_type', 'unknown')
        if hasattr(vision_output, 'last_hidden_state'):
            last_hidden = vision_output.last_hidden_state
            pooler = vision_output.pooler_output if hasattr(vision_output, 'pooler_output') else None
        elif hasattr(vision_output, 'pooler_output'):
            last_hidden = None
            pooler = vision_output.pooler_output
        else:
            last_hidden = vision_output if isinstance(vision_output, torch.Tensor) else None
            pooler = None

        if bridge_type in ['linear_bridge', 'better_mlp', 'multi_token']:
            if pooler is not None:
                vision_embeddings = pooler
            elif last_hidden is not None:
                vision_embeddings = last_hidden[:, 0, :]
            else:
                raise ValueError(f"Cannot extract vision embeddings for {bridge_type}")
        else:
            if last_hidden is not None and last_hidden.dim() == 3:
                vision_embeddings = last_hidden
            elif last_hidden is not None and last_hidden.dim() == 2:
                vision_embeddings = last_hidden.unsqueeze(1)
            elif pooler is not None:
                vision_embeddings = pooler.unsqueeze(1)
            else:
                raise ValueError(f"Cannot extract vision embeddings for {bridge_type}")

        vision_embeddings = vision_embeddings.detach()
        text_embeddings = self.model.language_model.model.embed_tokens(input_ids)
        text_embeddings = text_embeddings.to(dtype=model_dtype, device=self.device)

        if bridge_type == 'qformer':
            bridge_output = self.model.bridge(vision_embeddings, text_embeddings)
        else:
            bridge_output = self.model.bridge(vision_embeddings)
        if bridge_output.dim() == 2:
            bridge_output = bridge_output.unsqueeze(1)

        combined_embeddings = torch.cat([bridge_output, text_embeddings], dim=1)
        vision_attention = torch.ones(
            bridge_output.shape[0],
            bridge_output.shape[1],
            device=self.device,
            dtype=attention_mask.dtype
        )
        combined_attention_mask = torch.cat([vision_attention, attention_mask], dim=1)

        outputs = self.model.language_model.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            max_new_tokens=50,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            top_p=1.0,
            temperature=1.0
        )

        generated_ids = outputs[0]
        prompt_len = input_ids.shape[1]
        output_ids = generated_ids[prompt_len:] if generated_ids.shape[0] > prompt_len else generated_ids
        model_output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return model_output if model_output else "[Empty output]"

    @torch.no_grad()
    def _compute_epoch_text_metrics(self, epoch: int) -> Dict[str, float]:
        """Compute ref/ref1-aligned text metrics on full validation dataset."""
        import numpy as np
        from metrics.vqa_metrics import (
            BLEUScore, METEORScore, ROUGEScore, CIDErScore,
            PrecisionRecallF1, ExactMatchAccuracy, WUPS
        )

        if not hasattr(self, 'val_dataset') or len(self.val_dataset) == 0:
            return {}

        all_ground_truths: List[List[str]] = []
        all_generations: List[str] = []
        per_sample_records = []

        self.model.eval()
        pbar = tqdm(range(len(self.val_dataset)), desc=f"Epoch {epoch + 1} metrics", leave=False)
        for idx in pbar:
            sample = self.val_dataset[idx]
            question = sample.question if hasattr(sample, 'question') else 'N/A'
            gt_answers = self._extract_sample_answers(sample)

            try:
                generation = self._generate_answer_for_sample(sample)
            except Exception as e:
                generation = f"[Generation error: {str(e)[:80]}]"

            all_ground_truths.append(gt_answers)
            all_generations.append(generation)
            per_sample_records.append({
                'index': idx,
                'question': question,
                'prediction': generation,
                'ground_truths': gt_answers
            })

        bleu_metric = BLEUScore(n_gram=4)
        meteor_metric = METEORScore()
        rouge_metric = ROUGEScore(rouge_type='rougeL')
        cider_metric = CIDErScore(n_gram=4)
        prf_metric = PrecisionRecallF1()
        exact_match_metric = ExactMatchAccuracy(normalize=True)

        bleu_metric.update(all_generations, all_ground_truths)
        meteor_metric.update(all_generations, all_ground_truths)
        rouge_metric.update(all_generations, all_ground_truths)
        cider_metric.update(all_generations, all_ground_truths)
        prf_metric.update(all_generations, all_ground_truths)
        exact_match_metric.update(all_generations, all_ground_truths)

        bleu_result = bleu_metric.compute()
        meteor_result = meteor_metric.compute()
        rouge_result = rouge_metric.compute()
        cider_result = cider_metric.compute()
        prf_result = prf_metric.compute()
        exact_match_result = exact_match_metric.compute()

        # Simple accuracy (case-insensitive string match)
        simple_accuracy_scores = []
        for pred, refs in zip(all_generations, all_ground_truths):
            pred_lower = pred.lower().strip()
            match = any(pred_lower == ref.lower().strip() for ref in refs)
            simple_accuracy_scores.append(1.0 if match else 0.0)
        simple_accuracy = float(np.mean(simple_accuracy_scores)) if simple_accuracy_scores else 0.0

        # WUPS is available in vqa_metrics but not part of default ref/ref1 validation bundle.
        # We still compute it for side-by-side comparison requested in this repo.
        wups_scores = []
        try:
            wups_metric = WUPS(threshold=0.9)
            for pred, refs in zip(all_generations, all_ground_truths):
                best = 0.0
                for ref in refs:
                    sim = wups_metric._wup_similarity(pred.lower(), ref.lower())
                    score = wups_metric._threshold_wups(sim)
                    best = max(best, score)
                wups_scores.append(best)
            wups_avg = float(np.mean(wups_scores)) if wups_scores else 0.0
        except Exception as e:
            logger.warning(f"WUPS computation failed: {e}")
            wups_avg = 0.0
            wups_scores = []

        avg_metrics = {
            'accuracy': simple_accuracy,
            'exact_match': float(exact_match_result.value),
            'bleu': float(bleu_result.value),
            'rouge_l': float(rouge_result.value),
            'meteor': float(meteor_result.value),
            'cider': float(cider_result.value),
            'precision': float(prf_result.metadata.get('precision', 0.0)),
            'recall': float(prf_result.metadata.get('recall', 0.0)),
            'f1': float(prf_result.value),
            'wups': wups_avg,
        }

        results_dir = Path(self.config.output_dir) / "results"
        metrics_file = results_dir / f"text_metrics_epoch_{epoch + 1}.json"
        samples_file = results_dir / f"text_predictions_epoch_{epoch + 1}.json"
        aggregate_file = results_dir / "text_metrics_all_epochs.jsonl"

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                'epoch': epoch + 1,
                'num_samples': len(all_generations),
                'averages': avg_metrics,
                'details': {
                    'bleu': {'value': float(bleu_result.value), 'metadata': bleu_result.metadata},
                    'meteor': {'value': float(meteor_result.value), 'per_sample': meteor_result.per_sample},
                    'rouge_l': {'value': float(rouge_result.value), 'per_sample': rouge_result.per_sample},
                    'cider': {'value': float(cider_result.value), 'per_sample': cider_result.per_sample},
                    'exact_match': {'value': float(exact_match_result.value), 'per_sample': exact_match_result.per_sample},
                    'precision_recall_f1': {'value': float(prf_result.value), 'metadata': prf_result.metadata},
                    'wups@0.9': {'value': wups_avg, 'per_sample': wups_scores},
                }
            }, f, ensure_ascii=False, indent=2)

        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump({
                'epoch': epoch + 1,
                'samples': per_sample_records
            }, f, ensure_ascii=False, indent=2)

        with open(aggregate_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'epoch': epoch + 1,
                'num_samples': len(all_generations),
                **avg_metrics
            }, ensure_ascii=False) + "\n")

        logger.info("\n" + "=" * 80)
        logger.info(f"Epoch {epoch + 1} Text Metrics")
        logger.info("=" * 80)
        logger.info(f"  Accuracy:        {avg_metrics.get('accuracy', 0.0):.4f}")
        logger.info(f"  Exact Match:     {avg_metrics.get('exact_match', 0.0):.4f}")
        logger.info(f"  BLEU:            {avg_metrics.get('bleu', 0.0):.4f}")
        logger.info(f"  ROUGE-L:         {avg_metrics.get('rouge_l', 0.0):.4f}")
        logger.info(f"  METEOR:          {avg_metrics.get('meteor', 0.0):.4f}")
        logger.info(f"  CIDEr:           {avg_metrics.get('cider', 0.0):.4f}")
        logger.info(f"  Precision:       {avg_metrics.get('precision', 0.0):.4f}")
        logger.info(f"  Recall:          {avg_metrics.get('recall', 0.0):.4f}")
        logger.info(f"  F1:              {avg_metrics.get('f1', 0.0):.4f}")
        logger.info(f"  WUPS@0.9:        {avg_metrics.get('wups', 0.0):.4f}")
        logger.info(f"  Saved:      {metrics_file}")
        logger.info(f"  Samples:    {samples_file}")
        logger.info("=" * 80)

        return avg_metrics
    
    def _sample_inference(self, epoch: int, num_samples: int = 3):
        """Generate sample outputs on random validation samples."""
        import random
        
        if not hasattr(self, 'val_dataset') or len(self.val_dataset) == 0:
            return
        
        try:
            # Select random samples
            indices = random.sample(range(len(self.val_dataset)), min(num_samples, len(self.val_dataset)))
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Sample Inference - Epoch {epoch + 1}")
            logger.info(f"{'='*80}")
            
            self.model.eval()
            
            for i, idx in enumerate(indices, 1):
                sample = self.val_dataset[idx]
                
                # Get question and answer
                question = sample.question if hasattr(sample, 'question') else 'N/A'
                # Handle answers list (new schema) - use majority vote like collator does
                if hasattr(sample, 'answers'):
                    sample_answers = sample.answers if isinstance(sample.answers, list) else [sample.answers]
                    if len(sample_answers) == 0:
                        answer = 'N/A'
                    elif len(sample_answers) == 1:
                        answer = sample_answers[0]
                    else:
                        # Majority vote: get most common answer (same as collator)
                        from collections import Counter
                        counter = Counter(sample_answers)
                        answer = counter.most_common(1)[0][0]
                else:
                    answer = 'N/A'
                
                # Try to generate model output
                try:
                    model_output = self._generate_answer_for_sample(sample)
                except Exception as e:
                    model_output = f"[Generation error: {str(e)[:80]}]"
                
                # Log output
                logger.info(f"\n[Sample {i}]")
                logger.info(f"Question: {question}")
                logger.info(f"Model Output: {model_output}")
                logger.info(f"Ground truth: {answer}")
        
        except Exception as e:
            logger.warning(f"Sample inference failed: {e}")
        
        finally:
            self.model.train()
    
    def train(self):
        """Main training loop."""
        start_time = datetime.now()
        
        try:
            for epoch in range(self.config.num_epochs):
                epoch_start = datetime.now()
                
                # Train one epoch
                avg_loss, should_stop = self.train_epoch(epoch)
                
                # Run validation at end of epoch
                val_loss = self.validate()
                
                # Compute metrics
                epoch_elapsed = (datetime.now() - epoch_start).total_seconds()
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.config.learning_rate
                text_metrics = self._compute_epoch_text_metrics(epoch)
                
                # Log epoch summary with metrics
                epoch_log = []
                epoch_log.append(f"\n{'='*80}")
                epoch_log.append(f"Epoch {epoch+1}/{self.config.num_epochs}")
                epoch_log.append(f"{'='*80}")
                epoch_log.append(f"  Train Loss:        {avg_loss:.4f}")
                epoch_log.append(f"  [VAL] Val Loss:    {val_loss:.4f}")
                epoch_log.append(f"  Perplexity:        {perplexity:.4f}")
                epoch_log.append(f"  Learning Rate:     {current_lr:.2e}")
                epoch_log.append(f"  Early Stop Counter: {self.early_stop_counter}/{self.config.patience}")
                epoch_log.append(f"  Time:              {epoch_elapsed:.1f}s")
                epoch_log.append(f"  Best Val Loss:     {self.best_val_loss:.4f}")
                
                if text_metrics:
                    epoch_log.append("  Text Metrics (ref/ref1-aligned):")
                    epoch_log.append(f"    - BLEU:         {text_metrics.get('bleu', 0.0):.4f}")
                    epoch_log.append(f"    - METEOR:       {text_metrics.get('meteor', 0.0):.4f}")
                    epoch_log.append(f"    - ROUGE-L:      {text_metrics.get('rouge_l', 0.0):.4f}")
                    epoch_log.append(f"    - CIDEr:        {text_metrics.get('cider', 0.0):.4f}")
                    epoch_log.append(f"    - Exact Match:  {text_metrics.get('exact_match', 0.0):.4f}")
                    epoch_log.append(f"    - F1:           {text_metrics.get('f1', 0.0):.4f}")
                    epoch_log.append(f"    - WUPS@0.9:     {text_metrics.get('wups', 0.0):.4f}")
                
                # Log and save to file
                for line in epoch_log:
                    logger.info(line)
                    self._log_to_file(line)
                
                # Save epoch results to CSV
                is_best = val_loss < self.best_val_loss - self.config.min_delta
                epoch_metrics = {
                    'train_loss': avg_loss,
                    'val_loss': val_loss,
                    'perplexity': perplexity,
                    'learning_rate': current_lr,
                    'metric_bleu': text_metrics.get('bleu', 0.0),
                    'metric_meteor': text_metrics.get('meteor', 0.0),
                    'metric_rouge_l': text_metrics.get('rouge_l', 0.0),
                    'metric_cider': text_metrics.get('cider', 0.0),
                    'metric_exact_match': text_metrics.get('exact_match', 0.0),
                    'metric_precision': text_metrics.get('precision', 0.0),
                    'metric_recall': text_metrics.get('recall', 0.0),
                    'metric_f1': text_metrics.get('f1', 0.0),
                    'metric_wups': text_metrics.get('wups', 0.0),
                    'is_best': is_best,
                    'time_seconds': epoch_elapsed
                }
                self._save_epoch_results(epoch, epoch_metrics)
                
                # Show sample inference
                self._sample_inference(epoch, num_samples=3)
                
                if should_stop:
                    logger.warning(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Save final model
            if self.config.save_best and self.best_model_path:
                logger.info(f"✓ Best model at: {self.best_model_path}")
        
        except Exception as e:
            logger.error(f"✗ Training failed: {e}")
            raise
        
        finally:
            elapsed = datetime.now() - start_time
            total_hours = elapsed.total_seconds() / 3600
            logger.info(f"Training completed in {elapsed} ({total_hours:.2f}h)")
            
            # Save final summary
            self._save_final_summary(elapsed, total_hours)


# Backward compatibility
BridgeFineTuner = BridgeTrainer
