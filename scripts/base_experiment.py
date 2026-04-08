"""
Base experiment class for all ablation studies.
Standardizes model loading, data loading, and training setup.
Includes device auto-detection and optimization.
"""
import os
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

# Disable meta device to prevent meta tensor issues
os.environ["TRANSFORMERS_NO_META_DEVICE"] = "1"

from transformers import AutoModel
from src.schema.data_schema import OneSample
from src.middleware.logger import data_loader_logger
from utils.data_loader_helper import AblationDataLoader
from utils.device_detector import DeviceDetector


def is_kaggle() -> bool:
    """Check if running on Kaggle platform."""
    return os.path.exists('/kaggle/working')


class ExperimentConfig:
    """Base configuration for experiments."""
    
    # Model
    base_model_name: str = "5CD-AI/Vintern-1B-v3_5"
    torch_dtype: torch.dtype = torch.bfloat16
    low_cpu_mem_usage: bool = False
    use_flash_attn: bool = False
    
    # Training
    num_epochs: int = 10
    batch_size: int = 2  # Memory-optimized for 14GB GPU
    gradient_accumulation_steps: int = 4  # Effective batch size = 2 * 4 = 8
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
    
    def __init__(self, config: ExperimentConfig, auto_optimize: bool = True):
        """
        Initialize experiment with optional device auto-optimization.
        
        Args:
            config: ExperimentConfig instance
            auto_optimize: If True, auto-detect GPU and optimize training config
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Auto-detect and optimize for available GPU memory
        if auto_optimize:
            try:
                detector = DeviceDetector()
                detector.print_info()
                
                # Override config with device-optimized values
                device_config = detector.get_training_config()
                self.config.batch_size = device_config['batch_size']
                self.config.gradient_accumulation_steps = device_config['gradient_accumulation_steps']
                self.config.eval_steps = device_config['eval_steps']
                self.config.save_steps = device_config['save_steps']
                self.config.use_flash_attn = device_config['use_flash_attn']
                
                data_loader_logger.info(f"✓ Auto-optimized config for {self.device}")
                data_loader_logger.info(f"  Batch size: {self.config.batch_size}")
                data_loader_logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
                
            except Exception as e:
                data_loader_logger.warning(f"Auto-optimization failed: {e}, using default config")
        
        self.model = None
        self.train_samples: List[OneSample] = []
        self.val_samples: List[OneSample] = []
        self.test_samples: List[OneSample] = []
        
        data_loader_logger.info(f"Initializing experiment on device: {self.device}")
    
    def load_model(self) -> torch.nn.Module:
        """
        Load base model with transformers>=4.38.2 (has Qwen2Config for Vintern).
        TRANSFORMERS_NO_META_DEVICE=1 env var prevents meta tensor issues.
        """
        data_loader_logger.info(f"Loading base model: {self.config.base_model_name}")
        
        try:
            # Try with flash attention
            self.model = AutoModel.from_pretrained(
                self.config.base_model_name,
                torch_dtype=self.config.torch_dtype,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
                use_flash_attn=self.config.use_flash_attn,
            ).eval()
        except Exception as e:
            # Fallback without flash attention
            data_loader_logger.warning(f"Flash attention failed: {e}, retrying without it")
            self.model = AutoModel.from_pretrained(
                self.config.base_model_name,
                torch_dtype=self.config.torch_dtype,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
            ).eval()
        
        # Disable gradient checkpointing on frozen models (eliminates torch.utils.checkpoint warnings)
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        
        # Move to device
        self.model = self.model.to(self.device)
        data_loader_logger.info("Model loaded successfully")
        
        return self.model
    
    def load_data(
        self,
        max_samples: Optional[int] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        use_test_split: bool = True
    ) -> Tuple[List[OneSample], List[OneSample], List[OneSample]]:
        """Load training, validation, and test data."""
        data_loader_logger.info(
            f"Loading data (max_samples={max_samples}, "
            f"train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%})"
        )
        
        loader = AblationDataLoader()
        self.train_samples, self.val_samples, self.test_samples = loader.load_train_val_test_split(
            max_samples=max_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        ) if use_test_split else (
            *loader.load_train_val_split(
                max_samples=max_samples,
                val_ratio=val_ratio + test_ratio
            ),
            []
        )
        
        data_loader_logger.info(
            f"Data loaded: {len(self.train_samples)} train, "
            f"{len(self.val_samples)} val, {len(self.test_samples)} test"
        )
        
        return self.train_samples, self.val_samples, self.test_samples
    
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
    
    def run(
        self,
        max_samples: Optional[int] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        use_test_split: bool = True
    ):
        """Complete experiment pipeline."""
        self.print_config()
        
        # Load base model
        self.load_model()
        
        # Create fine-tuned model
        model = self.create_model()
        
        # Load data with optional test split
        self.load_data(
            max_samples=max_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            use_test_split=use_test_split
        )
        
        # Train
        self.train()
        
        print(f"\n✓ Experiment completed!")
        print(f"  Output: {self.config.output_dir}")
