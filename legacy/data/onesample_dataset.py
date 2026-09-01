"""
PyTorch Dataset wrapper for OneSample objects.
Allows using lists of OneSample objects directly with DataLoader.
"""
from typing import List, Union, Callable
import torch
from torch.utils.data import Dataset
from src.schema.data_schema import OneSample


class OneSampleDataset(Dataset):
    """
    PyTorch Dataset wrapper for lists of OneSample objects.
    
    This allows using OneSample lists directly with DataLoader:
    
    Example:
        samples = [OneSample(...), OneSample(...), ...]
        dataset = OneSampleDataset(samples)
        dataloader = DataLoader(
            dataset, 
            batch_size=8, 
            collate_fn=create_collate_fn()
        )
    """
    
    def __init__(
        self, 
        samples: List[OneSample],
        transform: Union[Callable, None] = None
    ):
        """
        Initialize dataset.
        
        Args:
            samples: List of OneSample objects
            transform: Optional transforms to apply to each sample
        """
        self.samples = samples
        self.transform = transform
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> OneSample:
        """
        Get a sample.
        
        Args:
            idx: Index of sample
        
        Returns:
            OneSample object (unchanged, to be processed by collate_fn)
        """
        sample = self.samples[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def to_dataloader(
        self,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,
        collate_fn: Union[Callable, None] = None,
        pin_memory: bool = True,
        **kwargs
    ):
        """
        Convenience method to create DataLoader directly from dataset.
        
        Example:
            dataset = OneSampleDataset(samples)
            dataloader = dataset.to_dataloader(
                batch_size=8,
                collate_fn=create_collate_fn()
            )
        """
        from torch.utils.data import DataLoader
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            **kwargs
        )
