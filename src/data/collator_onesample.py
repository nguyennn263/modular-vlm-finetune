"""
Custom collate function for OneSample objects.
Handles loading images, stacking tensors, and preserving text data as lists.
"""
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
from src.schema.data_schema import OneSample
from src.middleware.logger import data_loader_logger


def load_image(image_path: str, size: tuple = (336, 336)) -> torch.Tensor:
    """
    Load and convert image to tensor.
    
    Args:
        image_path: Path to image file
        size: Target image size (default: 336x336 for Vintern)
    
    Returns:
        Tensor of shape (3, H, W) in range [0, 1]
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(size, Image.Resampling.LANCZOS)
        # Convert to tensor: [0, 1] range
        image_tensor = torch.tensor(
            [*image.getdata()],
            dtype=torch.float32
        ).reshape(size[1], size[0], 3) / 255.0
        # Convert to (C, H, W) format
        image_tensor = image_tensor.permute(2, 0, 1)
        return image_tensor
    except Exception as e:
        data_loader_logger.warning(f"Failed to load image {image_path}: {e}")
        # Return black image as fallback
        return torch.zeros((3, size[1], size[0]), dtype=torch.float32)


def custom_collate_fn(
    batch: List[OneSample],
    image_size: tuple = (336, 336)
) -> Dict[str, Any]:
    """
    Custom collate function for batches of OneSample objects.
    
    Stacks images into tensors, keeps text data as lists for flexible processing.
    
    Args:
        batch: List of OneSample objects
        image_size: Target image size (H, W)
    
    Returns:
        Dictionary with batched data:
        - 'images': Stacked tensor of shape (B, 3, H, W)
        - 'questions': List of question strings
        - 'answers': List of answer strings
        - 'image_paths': List of image paths (for reference)
    """
    images = []
    questions = []
    answers = []
    image_paths = []
    
    for sample in batch:
        # Load and stack images
        image = load_image(sample.image_path, size=image_size)
        images.append(image)
        
        # Keep text data as lists
        questions.append(sample.question)
        answers.append(sample.answer)
        image_paths.append(sample.image_path)
    
    # Stack images into batch tensor
    batched_images = torch.stack(images, dim=0)  # (B, 3, H, W)
    
    return {
        "images": batched_images,
        "questions": questions,
        "answers": answers,
        "image_paths": image_paths,
    }


def create_collate_fn(image_size: tuple = (336, 336)):
    """
    Factory function to create a collate_fn with specific image size.
    
    Usage:
        collate_fn = create_collate_fn(image_size=(336, 336))
        dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    
    Args:
        image_size: Target image size (H, W)
    
    Returns:
        Collate function
    """
    def _collate(batch: List[OneSample]) -> Dict[str, Any]:
        return custom_collate_fn(batch, image_size=image_size)
    
    return _collate
