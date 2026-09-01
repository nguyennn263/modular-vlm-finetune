"""
Training data loader - unified interface for ablation studies and training benchmarks.
Handles both Kaggle and local environments automatically.
"""
from typing import List, Optional, Dict, Tuple
from pathlib import Path

from src.schema.data_schema import OneSample
from src.middleware.logger import data_loader_logger
from utils.data_loader_helper import AblationDataLoader
from utils.config_loader import load_config


class TrainingDataProvider:
    """
    Provides data for training with automatic environment detection.
    Caches loaded data to avoid repeated disk access.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.config = load_config(str(self.project_root / 'configs/data_configs.yaml'))
        self.data_loader = AblationDataLoader(str(self.project_root))
        self._cache: Dict[str, any] = {}
    
    def get_train_val_split(
        self,
        max_samples: Optional[int] = None,
        val_ratio: float = 0.1,
        use_cache: bool = True,
        cache_key: str = "default"
    ) -> Tuple[List[OneSample], List[OneSample]]:
        """
        Get train/validation split.
        
        Args:
            max_samples: Maximum samples to load
            val_ratio: Validation ratio (default: 0.1)
            use_cache: Cache results for reuse (default: True)
            cache_key: Cache identifier (default: 'default')
        
        Returns:
            Tuple of (train_samples, val_samples)
        """
        cache_id = f"{cache_key}_{max_samples}_{val_ratio}"
        
        if use_cache and cache_id in self._cache:
            data_loader_logger.info(f"Using cached data: {cache_id}")
            return self._cache[cache_id]
        
        train, val = self.data_loader.load_train_val_split(
            max_samples=max_samples,
            val_ratio=val_ratio
        )
        
        if use_cache:
            self._cache[cache_id] = (train, val)
        
        return train, val
    
    def get_raw_data(self, max_samples: Optional[int] = None) -> List[OneSample]:
        """Get all raw data without splitting."""
        return self.data_loader.load_raw_data(max_samples=max_samples)


def create_training_provider(project_root: str = ".") -> TrainingDataProvider:
    """Factory function to create training data provider."""
    return TrainingDataProvider(project_root)
