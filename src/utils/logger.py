import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import torch
from PIL import Image


class VLMLogger:
    """Logger cho VLM training với W&B support"""
    def __init__(
        self,
        project_name: str = "vietvlm-finetune",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        log_dir: str = "logs",
        use_wandb: bool = True,
    ):
        self.project_name = project_name
        self.run_name = run_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.wandb_run = None
        
        if use_wandb:
            self._init_wandb(config)
    
    def _init_wandb(self, config: Optional[Dict] = None):
        """Initialize W&B"""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.project_name,
                name=self.run_name,
                config=config,
            )
        except ImportError:
            print("Warning: wandb not installed. Logging locally only.")
            self.use_wandb = False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)
        
        # Also print
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Step {step}: {metrics_str}")
    
    def log_samples(
        self,
        images: List[Image.Image],
        questions: List[str],
        predictions: List[str],
        references: List[str],
        step: Optional[int] = None,
    ):
        """Log sample predictions với images"""
        if self.use_wandb and self.wandb_run:
            import wandb
            
            table = wandb.Table(columns=["Image", "Question", "Prediction", "Reference"])
            
            for img, q, pred, ref in zip(images, questions, predictions, references):
                table.add_data(wandb.Image(img), q, pred, ref)
            
            wandb.log({"samples": table}, step=step)
    
    def log_image(self, image: Image.Image, caption: str, step: Optional[int] = None):
        """Log single image"""
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log({"image": wandb.Image(image, caption=caption)}, step=step)
    
    def finish(self):
        """Finish logging"""
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()


def setup_logging(
    output_dir: str,
    project_name: str = "vietvlm-finetune",
    use_wandb: bool = True,
    config: Optional[Dict] = None,
) -> VLMLogger:
    """Setup logger"""
    return VLMLogger(
        project_name=project_name,
        log_dir=output_dir,
        use_wandb=use_wandb,
        config=config,
    )
