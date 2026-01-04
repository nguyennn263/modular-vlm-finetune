"""
Checkpoint utilities for VLM training
"""
import os
import re
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    step: int
    epoch: float
    loss: float
    eval_loss: Optional[float] = None
    eval_metrics: Optional[Dict[str, float]] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CheckpointMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CheckpointManager:
    """
    Manages checkpoint saving, loading, and cleanup for VLM training.
    
    Features:
    - Save/load model, optimizer, scheduler states
    - Track best checkpoint based on metric
    - Auto-cleanup old checkpoints
    - Resume from latest or specific checkpoint
    """
    
    def __init__(
        self,
        output_dir: str,
        save_total_limit: int = 3,
        save_best: bool = True,
        metric_for_best: str = "eval_loss",
        greater_is_better: bool = False,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_total_limit = save_total_limit
        self.save_best = save_best
        self.metric_for_best = metric_for_best
        self.greater_is_better = greater_is_better
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        self.best_metric = float('-inf') if greater_is_better else float('inf')
        self.best_checkpoint_path = None
        
        # Load existing best checkpoint info
        self._load_best_checkpoint_info()
    
    def _load_best_checkpoint_info(self):
        """Load best checkpoint info from disk"""
        best_info_path = self.checkpoint_dir / "best_checkpoint.json"
        if best_info_path.exists():
            with open(best_info_path) as f:
                info = json.load(f)
                self.best_metric = info.get("metric_value", self.best_metric)
                self.best_checkpoint_path = info.get("checkpoint_path")
    
    def _save_best_checkpoint_info(self):
        """Save best checkpoint info to disk"""
        best_info_path = self.checkpoint_dir / "best_checkpoint.json"
        with open(best_info_path, "w") as f:
            json.dump({
                "metric_name": self.metric_for_best,
                "metric_value": self.best_metric,
                "checkpoint_path": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            }, f, indent=2)
    
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        step: int,
        epoch: float,
        loss: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        eval_metrics: Optional[Dict[str, float]] = None,
        is_lora: bool = False,
    ) -> str:
        """
        Save a checkpoint.
        """
        from datetime import datetime
        
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if is_lora and hasattr(model, 'llm'):
            # Save LoRA adapter
            if hasattr(model.llm, 'save_pretrained'):
                model.llm.save_pretrained(checkpoint_path / "lora_adapter")
            # Save full model config
            model_path = checkpoint_path / "model"
            model_path.mkdir(exist_ok=True)
            torch.save({
                'vision_encoder_state': model.vision_encoder.state_dict() if hasattr(model, 'vision_encoder') else None,
                'projector_state': model.projector.state_dict() if hasattr(model, 'projector') else None,
            }, model_path / "non_lora_weights.pt")
        else:
            # Save full model
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(checkpoint_path / "model")
            else:
                torch.save(model.state_dict(), checkpoint_path / "model" / "pytorch_model.bin")
        
        # Save tokenizer
        tokenizer.save_pretrained(checkpoint_path / "tokenizer")
        
        # Save optimizer state
        if self.save_optimizer and optimizer is not None:
            torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        
        # Save scheduler state
        if self.save_scheduler and scheduler is not None:
            torch.save(scheduler.state_dict(), checkpoint_path / "scheduler.pt")
        
        # Save metadata
        metadata = CheckpointMetadata(
            step=step,
            epoch=epoch,
            loss=loss,
            eval_loss=eval_metrics.get("eval_loss") if eval_metrics else None,
            eval_metrics=eval_metrics,
            timestamp=datetime.now().isoformat(),
        )
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Update best checkpoint if needed
        if self.save_best and eval_metrics:
            current_metric = eval_metrics.get(self.metric_for_best)
            if current_metric is not None:
                is_better = (
                    (self.greater_is_better and current_metric > self.best_metric) or
                    (not self.greater_is_better and current_metric < self.best_metric)
                )
                if is_better:
                    self.best_metric = current_metric
                    self.best_checkpoint_path = checkpoint_path
                    self._save_best_checkpoint_info()
                    
                    # Create symlink to best checkpoint
                    best_link = self.checkpoint_dir / "best"
                    if best_link.exists() or best_link.is_symlink():
                        best_link.unlink()
                    best_link.symlink_to(checkpoint_path.name)
                    print(f"New best checkpoint! {self.metric_for_best}: {current_metric:.4f}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only save_total_limit"""
        checkpoints = self.list_checkpoints()
        
        # Always keep best checkpoint
        best_path = str(self.best_checkpoint_path) if self.best_checkpoint_path else None
        
        # Sort by step (newest first)
        checkpoints = sorted(checkpoints, key=lambda x: x["step"], reverse=True)
        
        # Keep only save_total_limit checkpoints
        to_delete = []
        kept = 0
        for ckpt in checkpoints:
            if ckpt["path"] == best_path:
                continue  # Always keep best
            if kept < self.save_total_limit:
                kept += 1
            else:
                to_delete.append(ckpt["path"])
        
        for path in to_delete:
            if Path(path).exists():
                shutil.rmtree(path)
                print(f"  Removed old checkpoint: {path}")
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints"""
        checkpoints = []
        
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                metadata_path = item / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    checkpoints.append({
                        "path": str(item),
                        "step": metadata.get("step", 0),
                        "epoch": metadata.get("epoch", 0),
                        "loss": metadata.get("loss"),
                        "eval_loss": metadata.get("eval_loss"),
                    })
        
        return sorted(checkpoints, key=lambda x: x["step"])
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint"""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[-1]["path"]
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get the path to the best checkpoint"""
        if self.best_checkpoint_path and Path(self.best_checkpoint_path).exists():
            return str(self.best_checkpoint_path)
        
        best_link = self.checkpoint_dir / "best"
        if best_link.exists():
            return str(best_link.resolve())
        
        return None
    
    def load_checkpoint(
        self,
        model: PreTrainedModel,
        checkpoint_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        is_lora: bool = False,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load model
        if is_lora and hasattr(model, 'llm'):
            # Load LoRA adapter
            lora_path = checkpoint_path / "lora_adapter"
            if lora_path.exists():
                from peft import PeftModel
                if not isinstance(model.llm, PeftModel):
                    # Re-apply LoRA config if needed
                    pass
                model.llm.load_adapter(str(lora_path), adapter_name="default")
            
            # Load non-LoRA weights
            non_lora_path = checkpoint_path / "model" / "non_lora_weights.pt"
            if non_lora_path.exists():
                weights = torch.load(non_lora_path, map_location="cpu")
                if weights.get("vision_encoder_state") and hasattr(model, 'vision_encoder'):
                    model.vision_encoder.load_state_dict(weights["vision_encoder_state"])
                if weights.get("projector_state") and hasattr(model, 'projector'):
                    model.projector.load_state_dict(weights["projector_state"])
        else:
            # Load full model
            model_path = checkpoint_path / "model"
            if (model_path / "pytorch_model.bin").exists():
                model.load_state_dict(torch.load(model_path / "pytorch_model.bin", map_location="cpu"))
            elif hasattr(model, 'load_pretrained'):
                model.load_pretrained(str(model_path))
        
        # Load optimizer
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer is not None and optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
            print("  ✓ Loaded optimizer state")
        
        # Load scheduler
        scheduler_path = checkpoint_path / "scheduler.pt"
        if scheduler is not None and scheduler_path.exists():
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
            print("  ✓ Loaded scheduler state")
        
        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        
        print(f"Resumed from step {metadata.get('step', 'unknown')}, epoch {metadata.get('epoch', 'unknown')}")
        
        return metadata
    
    def resolve_checkpoint_path(self, checkpoint: Optional[str]) -> Optional[str]:
        """
        Resolve checkpoint path from various formats.
        
        Args:
            checkpoint: Can be:
                - None: No checkpoint
                - "latest": Get latest checkpoint
                - "best": Get best checkpoint
                - Path to specific checkpoint
                
        Returns:
            Resolved checkpoint path or None
        """
        if checkpoint is None:
            return None
        
        if checkpoint == "latest":
            return self.get_latest_checkpoint()
        
        if checkpoint == "best":
            return self.get_best_checkpoint()
        
        # Check if it's a valid path
        if Path(checkpoint).exists():
            return checkpoint
        
        # Check if it's just a checkpoint name
        potential_path = self.checkpoint_dir / checkpoint
        if potential_path.exists():
            return str(potential_path)
        
        print(f"Warning: Checkpoint not found: {checkpoint}")
        return None


class CheckpointCallback(TrainerCallback):
    """
    Custom callback for HuggingFace Trainer to handle checkpoints.
    """
    
    def __init__(self, checkpoint_manager: CheckpointManager, is_lora: bool = False):
        self.checkpoint_manager = checkpoint_manager
        self.is_lora = is_lora
    
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when Trainer saves a checkpoint"""
        # Trainer handles saving, we just update our tracking
        pass
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called after evaluation"""
        metrics = kwargs.get("metrics", {})
        
        # Check if this is the best model
        if self.checkpoint_manager.save_best:
            metric_value = metrics.get(self.checkpoint_manager.metric_for_best)
            if metric_value is not None:
                is_better = (
                    (self.checkpoint_manager.greater_is_better and metric_value > self.checkpoint_manager.best_metric) or
                    (not self.checkpoint_manager.greater_is_better and metric_value < self.checkpoint_manager.best_metric)
                )
                if is_better:
                    self.checkpoint_manager.best_metric = metric_value
                    # Note: The actual saving is handled by Trainer's save logic
                    print(f"New best metric! {self.checkpoint_manager.metric_for_best}: {metric_value:.4f}")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in the output directory.
    """
    output_path = Path(output_dir)
    
    # Check HuggingFace Trainer checkpoints (checkpoint-XXXX format)
    trainer_checkpoints = list(output_path.glob("checkpoint-*"))
    
    # Check custom checkpoints
    custom_checkpoints = list((output_path / "checkpoints").glob("checkpoint-*")) if (output_path / "checkpoints").exists() else []
    
    all_checkpoints = trainer_checkpoints + custom_checkpoints
    
    if not all_checkpoints:
        return None
    
    # Sort by step number
    def get_step(path):
        match = re.search(r"checkpoint-(\d+)", path.name)
        return int(match.group(1)) if match else 0
    
    latest = max(all_checkpoints, key=get_step)
    return str(latest)


def auto_resume(output_dir: str, resume: Optional[str]) -> Optional[str]:
    """
    Automatically determine checkpoint to resume from.
    
    Args:
        output_dir: Output directory
        resume: User-specified resume option (None, "auto", "latest", "best", or path)
        
    Returns:
        Checkpoint path to resume from, or None
    """
    if resume is None:
        return None
    
    if resume == "auto":
        # Auto-detect if there's a checkpoint to resume from
        checkpoint = find_latest_checkpoint(output_dir)
        if checkpoint:
            print(f"Auto-resuming from: {checkpoint}")
        return checkpoint
    
    if resume == "latest":
        return find_latest_checkpoint(output_dir)
    
    if resume == "best":
        manager = CheckpointManager(output_dir)
        return manager.get_best_checkpoint()
    
    if Path(resume).exists():
        return resume
    
    # Try to find in output_dir
    potential_paths = [
        Path(output_dir) / resume,
        Path(output_dir) / "checkpoints" / resume,
    ]
    for path in potential_paths:
        if path.exists():
            return str(path)
    
    print(f"Warning: Could not find checkpoint: {resume}")
    return None
