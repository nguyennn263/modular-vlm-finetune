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
    
    # Distillation loss (critical for preventing embedding distribution collapse)
    use_distillation: bool = True
    distillation_loss_weight: float = 0.2  # λ for MSE loss: total = CE + λ*MSE
    warm_start: bool = True  # Initialize bridge from baseline weights
    
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
        if hasattr(self.model, 'baseline_bridge'):
            self.model.baseline_bridge = self.model.baseline_bridge.to(dtype=model_dtype)
        
        # Disable gradient checkpointing on all models (both top-level and nested modules)
        # This prevents checkpoint warnings since we're only training the bridge
        for module in [self.model.vision_model, self.model.language_model]:
            if module is not None:
                # Disable on top-level module
                if hasattr(module, 'gradient_checkpointing_disable'):
                    module.gradient_checkpointing_disable()
                # Also disable on all submodules (for nested architectures)
                for submodule in module.modules():
                    if hasattr(submodule, 'gradient_checkpointing'):
                        submodule.gradient_checkpointing = False
        
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
        
        # Warm start: initialize bridge from baseline if enabled
        if hasattr(self.model, 'warm_start_from_baseline') and self.config.warm_start:
            self.model.warm_start_from_baseline()
        
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
    
    def _setup_result_tracking(self):
        """Setup per-epoch result tracking and log files."""
        import csv
        from datetime import datetime
        
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
        except Exception as e:
            pass  # Silently fail if file write fails
    
    def _save_epoch_results(self, epoch: int, epoch_metrics: Dict):
        """Save per-epoch results to CSV."""
        import csv
        from datetime import datetime
        
        try:
            with open(self.epoch_results_file, 'a', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'epoch', 'global_step',
                        'train_loss', 'val_loss',
                        'perplexity', 'learning_rate',
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
                    'early_stop_counter': self.early_stop_counter,
                    'is_best': epoch_metrics.get('is_best', False),
                    'time_seconds': epoch_metrics.get('time_seconds', 0)
                })
        except Exception as e:
            logger.warning(f"Failed to save epoch results: {e}")
    
    def _save_final_summary(self, elapsed_timedelta, total_hours: float):
        """Save final training summary to JSON."""
        import json
        
        try:
            results_dir = Path(self.config.output_dir) / "results"
            summary_file = results_dir / "summary.json"
            
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
                    'summary': str(summary_file)
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"✓ Final summary saved to: {summary_file}")
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
        # - Patch-based (TileAttention, MiniQFormer, QFormer): full sequence [batch, num_patches, 1024]
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
        # Pooled-based bridges: residual, multi_token, gated_fusion (+ legacy names for compat)
        # Patch-based bridges: tile_attention, attention, mini_qformer, qformer
        pooled_bridges = ['residual', 'linear_bridge', 'multi_token', 'gated_fusion']
        if bridge_type in pooled_bridges:
            # Residual, MultiTokenMLP, GatedFusion expect single pooled vector [batch, 1024]
            # They expand to multiple tokens internally
            if pooler is not None:
                vision_embeddings = pooler
            elif last_hidden is not None:
                vision_embeddings = last_hidden[:, 0, :]  # Use CLS token
            else:
                vision_embeddings = vision_output
        else:
            # Patch-based bridges: tile_attention, attention, mini_qformer, qformer
            # These need full sequence [batch, num_patches, 1024]
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
        
        # NOTE: Do NOT detach vision embeddings during training!
        # Vision model is frozen (requires_grad=False), so gradients won't update it
        # Detach would break gradient flow to bridge, preventing training
        # vision_embeddings stays connected to computation graph for bridge training
        
        # Validate shapes before passing to bridge
        # 2D bridges expect pooled vectors [batch, dim]
        # 3D bridges expect patch sequences [batch, num_patches, dim]
        pooled_bridges_2d = ['residual', 'linear_bridge', 'multi_token', 'gated_fusion']
        patch_bridges_3d = ['tile_attention', 'mini_qformer', 'qformer']
        
        if bridge_type in pooled_bridges_2d:
            # These expect 2D pooled vectors [batch, 1024]
            assert vision_embeddings.dim() == 2, (
                f"Bridge {bridge_type} expects 2D vision_embeddings [batch, dim], "
                f"got shape {vision_embeddings.shape} (dim={vision_embeddings.dim()})"
            )
        elif bridge_type in patch_bridges_3d:
            # Patch-based bridges expect 3D sequences [batch, seq, dim]
            assert vision_embeddings.dim() == 3, (
                f"Bridge {bridge_type} expects 3D vision_embeddings [batch, seq, dim], "
                f"got shape {vision_embeddings.shape} (dim={vision_embeddings.dim()})"
            )
        else:
            raise ValueError(f"Unknown bridge type: {bridge_type}")
        
        # Get text embeddings early (needed for QFormer and concatenation)
        text_embeddings = self.model.language_model.model.embed_tokens(input_ids)
        # Convert to model dtype (embeddings are float32 by default)
        text_embeddings = text_embeddings.to(dtype=model_dtype)
        
        # NOTE: Do NOT detach text embeddings during training!
        # LLM is frozen (requires_grad=False), so gradients won't update it
        # Detach would break gradient flow to bridge
        # text_embeddings stays connected for full gradient computation
        
        # Apply bridge module (trainable)
        # Bridge handles both shape conversion and augmentation
        if bridge_type == 'qformer':
            # QFormer requires both vision features and question embeddings
            bridged_embeddings = self.model.bridge(vision_embeddings, text_embeddings)
        else:
            # All other bridges just take vision embeddings
            bridged_embeddings = self.model.bridge(vision_embeddings)
        
        # DEBUG: Verify gradients are flowing (log on first batch only)
        if self.global_step == 0 or self.global_step % 1000 == 0:
            bridge_params_trainable = sum(1 for p in self.model.bridge.parameters() if p.requires_grad)
            logger.info(f"[Gradient Check] vision_emb.requires_grad={vision_embeddings.requires_grad}, "
                       f"bridged_emb.requires_grad={bridged_embeddings.requires_grad}, "
                       f"bridge_trainable_params={bridge_params_trainable}")
        
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
        
        # Compute primary loss (next-token prediction) on ANSWER tokens only
        # logits has shape [batch_size, num_vision_tokens + text_len, vocab_size]
        # Skip all vision tokens and use only text logits
        text_logits = logits[:, num_vision_tokens:, :]  # [batch_size, text_len, vocab_size]
        
        # Shift for next-token prediction
        shift_logits = text_logits[..., :-1, :].contiguous()  # [batch_size, text_len-1, vocab_size]
        shift_labels = input_ids[..., 1:].contiguous()  # [batch_size, text_len-1]
        
        # IMPORTANT: Mask question tokens - only compute loss on Answer part
        # answer_start_pos tells us where Answer begins in the token stream
        # Set labels to -100 for question tokens (cross_entropy ignores -100)
        if 'answer_start_pos' in batch:
            answer_start_positions = batch['answer_start_pos']
            
            # Handle both single value and per-sample values
            if isinstance(answer_start_positions, torch.Tensor):
                # Could be [B] or scalar
                if answer_start_positions.dim() == 0:
                    # Scalar - same for all samples
                    answer_start_pos = answer_start_positions.item()
                    for i in range(shift_labels.shape[0]):
                        if answer_start_pos > 0:
                            shift_labels[i, :answer_start_pos-1] = -100
                else:
                    # Per-sample values [B]
                    for i in range(shift_labels.shape[0]):
                        answer_start_pos = answer_start_positions[i].item() if isinstance(answer_start_positions[i], torch.Tensor) else answer_start_positions[i]
                        if answer_start_pos > 0:
                            shift_labels[i, :answer_start_pos-1] = -100
            else:
                # Scalar int/float
                answer_start_pos = int(answer_start_positions)
                for i in range(shift_labels.shape[0]):
                    if answer_start_pos > 0:
                        shift_labels[i, :answer_start_pos-1] = -100
        
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction='mean'
        )
        
        # DEBUG: Log loss computation on first batch
        if self.global_step == 0 or self.global_step % 500 == 0:
            num_answer_tokens = (shift_labels != -100).sum().item()
            logger.info(f"[Loss] CE Loss computed on {num_answer_tokens} answer tokens "
                       f"(masked {(shift_labels == -100).sum().item()} question tokens)")
            if 'answer_start_pos' in batch:
                ans_pos = batch['answer_start_pos']
                if isinstance(ans_pos, torch.Tensor):
                    ans_pos = ans_pos.tolist() if ans_pos.dim() > 0 else [ans_pos.item()]
                logger.info(f"[Loss] answer_start_positions: {ans_pos}")
        
        # CRITICAL: Add distillation loss to prevent embedding collapse
        # Without this, model "hacks" the loss by producing out-of-distribution embeddings
        total_loss = ce_loss
        
        if self.config.use_distillation:
            # Get baseline (frozen) and bridge embeddings via dedicated method
            # Pass text_embeddings for QFormer which needs semantic context
            base_embeddings, bridge_embeddings = self.model.get_base_and_bridge_embeddings(
                pixel_values, 
                text_embeddings=text_embeddings
            )
            
            # Handle shape mismatches between different bridge architectures
            # Patch-based bridges (TileAttention, MiniQFormer, QFormer) may output different token counts
            # Solution: Pool both to (B, 1, 896) for fair semantic comparison
            
            # Pool base embeddings if needed
            if base_embeddings.dim() == 3 and base_embeddings.shape[1] > 1:
                # Multiple tokens: pool across sequence dimension
                base_embeddings = base_embeddings.mean(dim=1, keepdim=True)
            
            # Pool bridge embeddings if needed
            if bridge_embeddings.dim() == 4:
                # 4D tensor (multi-token output): flatten and pool
                # (B, 1, num_tokens, 896) → (B, 1, 896)
                bridge_embeddings = bridge_embeddings.mean(dim=2)
            elif bridge_embeddings.dim() == 3 and bridge_embeddings.shape[1] > 1:
                # 3D tensor with multiple tokens: pool across sequence
                # (B, num_tokens, 896) → (B, 1, 896)
                bridge_embeddings = bridge_embeddings.mean(dim=1, keepdim=True)
            
            # Ensure both are now (B, 1, 896)
            if base_embeddings.dim() == 2:
                base_embeddings = base_embeddings.unsqueeze(1)
            if bridge_embeddings.dim() == 2:
                bridge_embeddings = bridge_embeddings.unsqueeze(1)
            
            # MSE loss: keep bridge embeddings close to baseline distribution
            # This forces the bridge to work within the manifold the LLM understands
            distill_loss = F.mse_loss(bridge_embeddings, base_embeddings, reduction='mean')
            
            # Combine losses: CE + λ * MSE
            # λ typically 0.1-1.0 (starts with 0.5)
            # As training progresses, can reduce λ to allow more deviation
            total_loss = ce_loss + self.config.distillation_loss_weight * distill_loss
            
            # Log distillation metrics periodically
            if self.global_step % 500 == 0:
                logger.info(f"[Step {self.global_step}] CE_loss={ce_loss.item():.4f}, "
                           f"Distill_loss={distill_loss.item():.4f}, "
                           f"λ={self.config.distillation_loss_weight}")
        
        return total_loss
    
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
                    logger.info(f"Step {self.global_step}: train_loss={accumulated_loss:.4f}, [VAL] val_loss={val_loss:.4f}")
                    
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
                                answer = sample.answer if hasattr(sample, 'answer') else 'N/A'
                                
                                # Try to get model prediction using notebook approach
                                try:
                                    # Load image using dynamic preprocessing from notebook
                                    pixel_values = load_image(
                                        sample.image_path, 
                                        input_size=448, 
                                        max_num=6
                                    ).to(self.model.device)
                                    
                                    # Get model dtype from vision model
                                    model_dtype = next(self.model.vision_model.parameters()).dtype
                                    pixel_values = pixel_values.to(dtype=model_dtype)
                                    
                                    # Prepare question with prompt format (from notebook)
                                    question_text = f"<image>\nQuestion: {question}\nAnswer:"
                                    
                                    # Generate answer - custom inference with bridge
                                    with torch.no_grad():
                                        # Get vision embeddings via bridge (respects training)
                                        # pixel_values shape: [num_patches, 3, 448, 448] - process first patch only
                                        if pixel_values.dim() == 4:
                                            pixel_values_input = pixel_values[0:1, :, :, :]  # [1, 3, 448, 448]
                                        else:
                                            pixel_values_input = pixel_values.unsqueeze(0)
                                        vision_output = self.model.vision_model(pixel_values_input)
                                        
                                        # Extract tensor from BaseModelOutputWithPooling using same logic as forward_pass
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
                                        if bridge_type in ['linear_bridge', '', 'multi_token']:
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
                                        bridged_embeddings = self.model.bridge(vision_embeddings)
                                        if bridged_embeddings.dim() == 2:
                                            bridged_embeddings = bridged_embeddings.unsqueeze(1)
                                        
                                        # Tokenize question
                                        inputs = self.tokenizer(
                                            question_text, 
                                            return_tensors='pt',
                                            max_length=256,
                                            truncation=True
                                        )
                                        input_ids = inputs['input_ids'].to(self.device)
                                        attention_mask = inputs['attention_mask'].to(self.device)
                                        
                                        # Get text embeddings
                                        text_embeddings = self.model.language_model.model.embed_tokens(input_ids)
                                        text_embeddings = text_embeddings.to(dtype=model_dtype)
                                        
                                        # Combine vision + text embeddings
                                        combined = torch.cat([bridged_embeddings, text_embeddings], dim=1)
                                        vision_attention = torch.ones(
                                            bridged_embeddings.shape[0], 
                                            bridged_embeddings.shape[1], 
                                            device=self.device, 
                                            dtype=attention_mask.dtype
                                        )
                                        combined_attention = torch.cat([vision_attention, attention_mask], dim=1)
                                        
                                        # Generate tokens greedily (200 token max)
                                        generated_tokens = []
                                        max_new_tokens = 200
                                        
                                        outputs = self.model.language_model(
                                            inputs_embeds=combined,
                                            attention_mask=combined_attention,
                                            return_dict=True
                                        )
                                        logits = outputs.logits
                                        
                                        for step in range(max_new_tokens):
                                            last_logits = logits[:, -1, :]
                                            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                                            generated_tokens.append(next_token)
                                            
                                            # Stop at EOS/PAD
                                            if next_token.item() == self.tokenizer.eos_token_id or next_token.item() == self.tokenizer.pad_token_id:
                                                break
                                            
                                            # Continue generation
                                            next_embedding = self.model.language_model.model.embed_tokens(next_token)
                                            next_embedding = next_embedding.to(dtype=model_dtype)
                                            combined = torch.cat([combined, next_embedding], dim=1)
                                            
                                            next_attention = torch.ones(
                                                1, 1, 
                                                device=self.device, 
                                                dtype=attention_mask.dtype
                                            )
                                            combined_attention = torch.cat([combined_attention, next_attention], dim=1)
                                            
                                            outputs = self.model.language_model(
                                                inputs_embeds=combined,
                                                attention_mask=combined_attention,
                                                return_dict=True
                                            )
                                            logits = outputs.logits
                                        
                                        # Decode answer
                                        if generated_tokens:
                                            generated_ids = torch.cat(generated_tokens, dim=1)
                                            model_output = self.tokenizer.decode(
                                                generated_ids[0], 
                                                skip_special_tokens=True
                                            ).strip()
                                        else:
                                            model_output = "[No answer generated]"
                                    
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
    
    def _sample_inference(self, epoch: int, num_samples: int = 3):
        """Generate sample outputs on random validation samples."""
        import random
        
        if not hasattr(self, 'val_dataset') or len(self.val_dataset) == 0:
            logger.debug("No validation dataset for sample inference")
            return
        
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            logger.warning("Tokenizer not initialized, skipping sample inference")
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
                answer = sample.answer if hasattr(sample, 'answer') else 'N/A'
                
                # Try to generate model output
                model_output = None
                try:
                    # Load image using notebook preprocessing
                    pixel_values = load_image(
                        sample.image_path, 
                        input_size=448, 
                        max_num=6
                    )
                    
                    # Convert to model dtype and move to device
                    model_dtype = next(self.model.vision_model.parameters()).dtype
                    pixel_values = pixel_values.to(dtype=model_dtype, device=self.device)
                    
                    # Tokenize question
                    question_text = f"Question: {question}\nAnswer:"
                    inputs = self.tokenizer(
                        question_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    input_ids = inputs['input_ids'].to(self.device)
                    
                    # Get bridge model's inference output
                    with torch.no_grad():
                        # Get vision embeddings
                        # pixel_values shape: [num_patches, 3, 448, 448] - process first patch for inference
                        if pixel_values.dim() == 4:
                            pixel_values_input = pixel_values[0:1, :, :, :]  # [1, 3, 448, 448]
                        else:
                            pixel_values_input = pixel_values.unsqueeze(0)
                        vision_output = self.model.vision_model(pixel_values_input)
                        
                        # Extract tensor from BaseModelOutputWithPooling using same logic as forward_pass
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
                        # Pooled-based bridges: residual, multi_token, gated_fusion (+ legacy names for compat)
                        # Patch-based bridges: tile_attention, attention, mini_qformer, qformer
                        pooled_bridges = ['residual', 'linear_bridge', 'multi_token', 'gated_fusion']
                        if bridge_type in pooled_bridges:
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
                        
                        # Apply bridge
                        bridge_output = self.model.bridge(vision_embeddings)
                        if bridge_output.dim() == 2:
                            bridge_output = bridge_output.unsqueeze(1)
                        
                        # Get text embeddings
                        text_embeddings = self.model.language_model.model.embed_tokens(input_ids)
                        
                        # Combine embeddings
                        combined_embeddings = torch.cat([bridge_output, text_embeddings], dim=1)
                        
                        # Generate with language model
                        outputs = self.model.language_model.generate(
                            inputs_embeds=combined_embeddings,
                            max_new_tokens=50,
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                        
                        # Decode output (skip input tokens)
                        output_ids = outputs[0, combined_embeddings.shape[1]:]
                        model_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    
                except Exception as e:
                    model_output = f"[Generation error: {str(e)[:80]}]"
                
                # Log output
                logger.info(f"\n[Sample {i}]")
                logger.info(f"Question: {question}")
                if model_output:
                    logger.info(f"Model Output: {model_output}")
                logger.info(f"Ground truth: {answer}")
        
        except Exception as e:
            import traceback
            logger.error(f"Sample inference failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
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
                
                # Log epoch summary with metrics
                logger.info(f"\n{'='*80}")
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
                logger.info(f"{'='*80}")
                logger.info(f"  Train Loss:        {avg_loss:.4f}")
                logger.info(f"  [VAL] Val Loss:    {val_loss:.4f}")
                logger.info(f"  Perplexity:        {perplexity:.4f}")
                logger.info(f"  Learning Rate:     {current_lr:.2e}")
                logger.info(f"  Early Stop Counter: {self.early_stop_counter}/{self.config.patience}")
                logger.info(f"  Time:              {epoch_elapsed:.1f}s")
                logger.info(f"  Best Val Loss:     {self.best_val_loss:.4f}")
                
                # Save epoch results to CSV
                is_best = val_loss < self.best_val_loss - self.config.min_delta
                epoch_metrics = {
                    'train_loss': avg_loss,
                    'val_loss': val_loss,
                    'perplexity': perplexity,
                    'learning_rate': current_lr,
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
