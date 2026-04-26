"""
Custom collate function for OneSample objects.
Handles loading images, tokenizing text, and creating batches for VLM training.
"""
import torch
import random
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
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
    answers_list = []
    
    for sample in batch:
        # Load and stack images
        image = load_image(sample.image_path, size=image_size)
        images.append(image)
        
        # Keep text data
        questions.append(sample.question)
        
        # Handle multiple answers: use majority vote like ref1/ does
        # If only one answer, use it. Otherwise use the most common one.
        sample_answers = sample.answers if sample.answers else ['']
        if len(sample_answers) == 1:
            selected_answer = sample_answers[0]
        else:
            # Majority vote: get most common answer
            counter = Counter(sample_answers)
            selected_answer = counter.most_common(1)[0][0]
        
        answers_list.append(selected_answer)
    
    # Stack images into batch tensor
    pixel_values = torch.stack(images, dim=0)  # (B, 3, H, W)
    
    result = {}
    result['pixel_values'] = pixel_values
    
    # Tokenize text if tokenizer is provided
    if tokenizer is not None:
        # Using official Vintern format (from ref2/conversation.py)
        system_message = "Bạn là một mô hình trí tuệ nhân tạo đa phương thức Tiếng Việt có tên gọi là Vintern, được phát triển bởi người Việt. Bạn là một trợ lý trí tuệ nhân tạo hữu ích và không gây hại."
        
        # Calculate where Answer begins for loss masking
        input_ids_list = []
        attention_mask_list = []
        answer_start_positions = []
        
        for q, a in zip(questions, answers_list):
            # Extract clean question (remove <image>\n if present)
            question_clean = q
            if q.startswith("<image>\n"):
                question_clean = q[8:]
            
            # Full text including answer
            full_text = (
                f"<|im_start|>system\n{system_message}<|im_end|>\n"
                f"<|im_start|>user\n<image>\n{question_clean}<|im_end|>\n"
                f"<|im_start|>assistant\n{a}<|im_end|>"
            )
            
            # Text up to assistant answer header (for masking)
            question_part = (
                f"<|im_start|>system\n{system_message}<|im_end|>\n"
                f"<|im_start|>user\n<image>\n{question_clean}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            # Encode both
            full_enc = tokenizer(
                full_text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            question_enc = tokenizer(
                question_part,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # answer_start_pos: where answer tokens begin
            answer_start_pos = question_enc['input_ids'].shape[1]
            
            input_ids_list.append(full_enc['input_ids'].squeeze(0))
            attention_mask_list.append(full_enc['attention_mask'].squeeze(0))
            answer_start_positions.append(answer_start_pos)
        
        # Pad to same length for batch
        max_seq_len = max(ids.shape[0] for ids in input_ids_list)
        input_ids_batch = []
        attention_mask_batch = []
        
        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_seq_len - ids.shape[0]
            if pad_len > 0:
                ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            input_ids_batch.append(ids)
            attention_mask_batch.append(mask)
        
        result['input_ids'] = torch.stack(input_ids_batch)
        result['attention_mask'] = torch.stack(attention_mask_batch)
        result['answer_start_pos'] = torch.tensor(answer_start_positions, dtype=torch.long)
    else:
        # Return raw text if no tokenizer
        result['questions'] = questions
        result['answers'] = answers_list
    
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
