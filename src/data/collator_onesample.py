"""
Custom collate function for OneSample objects.
Handles loading images, tokenizing text, and creating batches for VLM training.
"""
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional
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
    tokenizer: Optional[Any] = None,
    image_size: tuple = (336, 336),
    max_length: int = 512
) -> Dict[str, Any]:
    """
    Custom collate function for batches of OneSample objects.
    
    Loads images and tokenizes text for VLM training.
    
    Args:
        batch: List of OneSample objects
        tokenizer: Tokenizer for text (if None, returns raw text)
        image_size: Target image size (H, W)
        max_length: Maximum sequence length for tokenization
    
    Returns:
        Dictionary with batched data:
        - 'pixel_values': Stacked tensor of shape (B, 3, H, W)
        - 'input_ids': Tokenized text (if tokenizer provided)
        - 'attention_mask': Attention mask (if tokenizer provided)
        - Otherwise returns 'questions' and 'answers' as lists
    """
    images = []
    questions = []
    answers = []
    
    for sample in batch:
        # Load and stack images
        image = load_image(sample.image_path, size=image_size)
        images.append(image)
        
        # Keep text data
        questions.append(sample.question)
        answers.append(sample.answer)
    
    # Stack images into batch tensor
    pixel_values = torch.stack(images, dim=0)  # (B, 3, H, W)
    
    result = {}
    result['pixel_values'] = pixel_values
    
    # Tokenize text if tokenizer is provided
    if tokenizer is not None:
        # Format following Vintern's pattern: <image>\nQuestion prompt\nAnswer
        # This matches the format used in the inference notebook
        texts = [
            f"<image>\nQuestion: {q}\nAnswer: {a}"
            for q, a in zip(questions, answers)
        ]
        
        # Tokenize with padding and truncation
        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        result['input_ids'] = encoded['input_ids']
        result['attention_mask'] = encoded['attention_mask']
    else:
        # Return raw text if no tokenizer
        result['questions'] = questions
        result['answers'] = answers
    
    return result


def create_collate_fn(
    tokenizer: Optional[Any] = None,
    image_size: tuple = (336, 336),
    max_length: int = 512
):
    """
    Factory function to create a collate_fn with specific configuration.
    
    Usage:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('model_name')
        collate_fn = create_collate_fn(tokenizer=tokenizer)
        dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    
    Args:
        tokenizer: Tokenizer instance (if None, returns raw text)
        image_size: Target image size (H, W)
        max_length: Maximum sequence length
    
    Returns:
        Collate function
    """
    def _collate(batch: List[OneSample]) -> Dict[str, Any]:
        return custom_collate_fn(
            batch,
            tokenizer=tokenizer,
            image_size=image_size,
            max_length=max_length
        )
    
    return _collate
