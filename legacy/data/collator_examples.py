"""
Example: Using custom collate_fn with OneSample objects

Shows how to leverage the custom collate_fn to properly batch OneSample objects
with their images loaded and stacked into tensors.
"""

from torch.utils.data import DataLoader
from src.data.onesample_dataset import OneSampleDataset
from src.data.collator_onesample import create_collate_fn
from utils.data_loader_helper import load_ablation_data


def example_basic_dataloader():
    """
    Example 1: Create DataLoader with custom collate function
    """
    # Load data
    train_samples, val_samples = load_ablation_data(max_samples=100, val_ratio=0.1)
    
    # Create dataset wrapper
    train_dataset = OneSampleDataset(train_samples)
    
    # Create DataLoader with custom collate function
    collate_fn = create_collate_fn(image_size=(336, 336))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Iterate through batches
    for batch in train_loader:
        images = batch['images']  # Tensor (B, 3, 336, 336)
        questions = batch['questions']  # List of strings
        answers = batch['answers']  # List of strings
        
        print(f"Batch size: {images.shape[0]}")
        print(f"Image shape: {images.shape}")
        print(f"Questions: {questions}")
        print(f"Answers: {answers}")
        break


def example_convenience_method():
    """
    Example 2: Use OneSampleDataset.to_dataloader() convenience method
    """
    train_samples, val_samples = load_ablation_data(max_samples=100)
    
    # Create dataset and DataLoader in one step
    train_dataset = OneSampleDataset(train_samples)
    
    collate_fn = create_collate_fn(image_size=(336, 336))
    
    train_loader = train_dataset.to_dataloader(
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    for batch in train_loader:
        print(f"✓ Batch loaded with {len(batch['questions'])} samples")
        break


def example_with_trainer():
    """
    Example 3: Use with BridgeTrainer
    """
    from src.training.trainer import BridgeTrainer, TrainConfig
    
    # Load data
    train_samples, val_samples = load_ablation_data(max_samples=100)
    
    # Convert to Dataset
    train_dataset = OneSampleDataset(train_samples)
    val_dataset = OneSampleDataset(val_samples)
    
    # Trainer will automatically detect OneSample type and use custom collate_fn
    config = TrainConfig(batch_size=8, num_epochs=3)
    
    # Assuming you have a model...
    # trainer = BridgeTrainer(model, train_dataset, val_dataset, config)
    print("✓ Trainer will use custom collate_fn for OneSample objects")


if __name__ == "__main__":
    print("Example 1: Basic DataLoader with custom collate_fn")
    print("=" * 60)
    # example_basic_dataloader()
    
    print("\nExample 2: Convenience method")
    print("=" * 60)
    # example_convenience_method()
    
    print("\nExample 3: With BridgeTrainer")
    print("=" * 60)
    example_with_trainer()
