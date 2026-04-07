"""
Custom Collate Function for OneSample Objects

This module provides utilities for batching OneSample objects with PyTorch DataLoaders.

Key Components:
1. custom_collate_fn() - Core function that:
   - Loads images from image_path and stacks them into (B, 3, H, W) tensor
   - Keeps questions and answers as lists (flexible processing)
   - Returns structured batch dict

2. create_collate_fn() - Factory function for creating collate_fn with specific image size

3. OneSampleDataset - PyTorch Dataset wrapper for lists of OneSample objects

4. load_image() - Helper to load and normalize images


USAGE
=====

Option 1: Basic Usage
-------------------
from torch.utils.data import DataLoader
from src.data.onesample_dataset import OneSampleDataset
from src.data.collator_onesample import create_collate_fn
from utils.data_loader_helper import load_ablation_data

# Load data
train_samples, val_samples = load_ablation_data(max_samples=1000)

# Create dataset wrapper
train_dataset = OneSampleDataset(train_samples)

# Create DataLoader with custom collate
collate_fn = create_collate_fn(image_size=(336, 336))
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

# Iterate
for batch in train_loader:
    images = batch['images']        # (B, 3, 336, 336)
    questions = batch['questions']  # List[str]
    answers = batch['answers']      # List[str]


Option 2: Convenience Method
----------------------------
train_dataset = OneSampleDataset(train_samples)
train_loader = train_dataset.to_dataloader(
    batch_size=8,
    collate_fn=create_collate_fn()
)


Option 3: With BridgeTrainer
---------------------------
train_dataset = OneSampleDataset(train_samples)
val_dataset = OneSampleDataset(val_samples)

trainer = BridgeTrainer(
    model,
    train_dataset,
    val_dataset,
    config
)
# Trainer automatically detects OneSample type and applies custom collate_fn


BATCH OUTPUT FORMAT
===================

The collate_fn returns a dictionary with:

{
    'images': torch.Tensor,          # (batch_size, 3, height, width)
    'questions': List[str],          # Question texts
    'answers': List[str],            # Answer texts  
    'image_paths': List[str],        # Original image paths (for debugging)
}

All images are normalized to [0, 1] range and resized to the specified size.


CUSTOM IMAGE SIZE
=================

By default, images are resized to (336, 336) for Vintern-1B.

To use different size:
    collate_fn = create_collate_fn(image_size=(224, 224))


ERROR HANDLING
==============

If an image fails to load, a black fallback image is returned.
Check logs for details:
    data_loader_logger.warning(f"Failed to load image {image_path}: {e}")
"""

# Implementation note: See collator_onesample.py for full implementation
