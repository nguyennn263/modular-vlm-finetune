"""
High-level data loader helper for ablation studies and training.
Automatically detects environment and provides correct data paths.
"""
from pathlib import Path
from typing import List, Optional, Tuple

from src.schema.data_schema import OneSample
from src.middleware.logger import data_loader_logger
from src.data.unified_loader import UnifiedDataLoader
from src.data.environment import EnvironmentDetector
from utils.config_loader import load_config


class AblationDataLoader:
    """
    High-level data loader for ablation studies.
    Automatically handles Kaggle and local environments.
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize loader with automatic environment detection."""
        self.project_root = Path(project_root)
        self.data_config = load_config(str(self.project_root / 'configs/data_configs.yaml'))
        self.is_kaggle = EnvironmentDetector.is_kaggle()
        self.loader = UnifiedDataLoader(
            self.data_config,
            self.data_config['kaggle_setup'],
            str(self.project_root)
        )
    
    def load_raw_data(self, max_samples: Optional[int] = None) -> List[OneSample]:
        """
        Load raw data samples.
        
        Args:
            max_samples: Maximum samples to load (None = all)
        
        Returns:
            List of OneSample objects
        """
        return self.loader.load_raw_data(max_samples=max_samples)
    
    def load_train_val_split(
        self, 
        max_samples: Optional[int] = None,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[OneSample], List[OneSample]]:
        """
        Load data and split into train/validation.
        
        Args:
            max_samples: Maximum samples to load
            val_ratio: Validation set ratio (default: 0.1)
            seed: Random seed
        
        Returns:
            Tuple of (train_samples, val_samples)
        """
        import random
        
        samples = self.load_raw_data(max_samples=max_samples)
        
        # Shuffle with seed
        random.seed(seed)
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        # Split
        split_idx = int(len(shuffled) * (1 - val_ratio))
        train_samples = shuffled[:split_idx]
        val_samples = shuffled[split_idx:]
        
        data_loader_logger.info(
            f"Split data: {len(train_samples)} train, {len(val_samples)} val"
        )
        
        return train_samples, val_samples

    def load_train_val_test_split(
        self,
        max_samples: Optional[int] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[OneSample], List[OneSample], List[OneSample]]:
        """
        Load data and split into train/validation/test.
        
        Args:
            max_samples: Maximum samples to load
            train_ratio: Training set ratio (default: 0.8)
            val_ratio: Validation set ratio (default: 0.1)
            test_ratio: Test set ratio (default: 0.1)
            seed: Random seed
        
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        import random
        
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
        
        samples = self.load_raw_data(max_samples=max_samples)
        
        # Shuffle with seed
        random.seed(seed)
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        # Split
        train_idx = int(len(shuffled) * train_ratio)
        val_idx = train_idx + int(len(shuffled) * val_ratio)
        
        train_samples = shuffled[:train_idx]
        val_samples = shuffled[train_idx:val_idx]
        test_samples = shuffled[val_idx:]
        
        data_loader_logger.info(
            f"Split data: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test"
        )
        
        return train_samples, val_samples, test_samples


def load_ablation_data(
    max_samples: Optional[int] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    project_root: str = ".",
    use_test_split: bool = True
) -> Tuple[List[OneSample], List[OneSample], List[OneSample]]:
    """
    Convenience function to load train/val/test split for ablation studies.
    
    Args:
        max_samples: Maximum samples to load
        train_ratio: Training split ratio (default: 0.8)
        val_ratio: Validation split ratio (default: 0.1)
        test_ratio: Test split ratio (default: 0.1)
        project_root: Project root directory
        use_test_split: Whether to return 3-way split (default: True)
    
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    loader = AblationDataLoader(project_root)
    if use_test_split:
        return loader.load_train_val_test_split(
            max_samples=max_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
    else:
        # Legacy: only train/val split
        train, val = loader.load_train_val_split(
            max_samples=max_samples,
            val_ratio=val_ratio + test_ratio
        )
        return train, val, []
