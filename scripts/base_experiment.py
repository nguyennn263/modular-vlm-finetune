"""
Base experiment class for all ablation studies.
Standardizes model loading, data loading, and training setup.
"""
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from transformers import AutoModel
from src.schema.data_schema import OneSample
from src.middleware.logger import data_loader_logger
from utils.data_loader_helper import AblationDataLoader


class ExperimentConfig:
    """Base configuration for experiments."""
    
    # Model
    base_model_name: str = "5CD-AI/Vintern-1B-v3_5"
    torch_dtype: torch.dtype = torch.bfloat16
    low_cpu_mem_usage: bool = False
    use_flash_attn: bool = False
    
    # Training
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 2e-4
    eval_steps: int = 100
    save_steps: int = 500
    
    # Output
    output_dir: str = "checkpoints/exp_base"


class BaseExperiment(ABC):
    """
    Base class for all ablation experiments.
    Handles model loading, data loading, and common setup.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_samples: List[OneSample] = []
        self.val_samples: List[OneSample] = []
        
        data_loader_logger.info(f"Initializing experiment on device: {self.device}")
    
    def load_model(self) -> torch.nn.Module:
        """
        Load base model using notebook-compatible approach.
        Uses low_cpu_mem_usage=False and avoids device_map="auto".
        """
        data_loader_logger.info(f"Loading base model: {self.config.base_model_name}")
        
        try:
            # Try with flash attention
            self.model = AutoModel.from_pretrained(
                self.config.base_model_name,
                torch_dtype=self.config.torch_dtype,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage,
                trust_remote_code=True,
                use_flash_attn=self.config.use_flash_attn,
            ).eval()
        except Exception as e:
            # Fallback without specific attention implementation
            data_loader_logger.warning(f"Flash attention failed: {e}, retrying without it")
            self.model = AutoModel.from_pretrained(
                self.config.base_model_name,
                torch_dtype=self.config.torch_dtype,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage,
                trust_remote_code=True,
            ).eval()
        
        # Move to device (works on Kaggle and local)
        self.model = self.model.to(self.device)
        data_loader_logger.info("Model loaded successfully")
        
        return self.model
    
    def load_data(
        self,
        max_samples: Optional[int] = None,
        val_ratio: float = 0.1
    ) -> Tuple[List[OneSample], List[OneSample]]:
        """Load training and validation data."""
        data_loader_logger.info(f"Loading data (max_samples={max_samples}, val_ratio={val_ratio})")
        
        loader = AblationDataLoader()
        self.train_samples, self.val_samples = loader.load_train_val_split(
            max_samples=max_samples,
            val_ratio=val_ratio
        )
        
        data_loader_logger.info(
            f"Data loaded: {len(self.train_samples)} train, {len(self.val_samples)} val"
        )
        
        return self.train_samples, self.val_samples
    
    def print_config(self):
        """Print configuration."""
        print("=" * 80)
        print(f"EXPERIMENT: {self.__class__.__name__}")
        print("=" * 80)
        print(f"Model: {self.config.base_model_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Output dir: {self.config.output_dir}")
        print()
    
    @abstractmethod
    def create_model(self) -> torch.nn.Module:
        """Create fine-tuned model with bridge. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def train(self):
        """Run training. Must be implemented by subclass."""
        pass
    
    def run(self, max_samples: Optional[int] = None, val_ratio: float = 0.1):
        """Complete experiment pipeline."""
        self.print_config()
        
        # Load base model
        self.load_model()
        
        # Create fine-tuned model
        model = self.create_model()
        
        # Load data
        self.load_data(max_samples=max_samples, val_ratio=val_ratio)
        
        # Train
        self.train()
        
        print(f"\n✓ Experiment completed!")
        print(f"  Output: {self.config.output_dir}")
