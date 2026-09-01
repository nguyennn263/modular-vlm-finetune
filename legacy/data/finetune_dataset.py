"""
Fine-tuning Dataset for Vision-Language models.
Supports loading from JSON with image paths and text pairs.
"""

import os
from pathlib import Path
from typing import Union, List, Dict, Optional
import json

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer

from src.schema.data_schema import OneSample
from src.data.data_actions import load_dataset_from_json
from src.middleware.logger import data_loader_logger


class FineTuneDataset(Dataset):
    """
    Dataset for fine-tuning vision-language models.
    
    Loads OneSample objects (image_path, question, answer).
    Returns (image, text) pairs with tokenized text.
    """
    
    def __init__(self,
                 data: Union[str, List[OneSample], List[Dict]],
                 image_dir: str,
                 tokenizer_name: str = "Qwen/Qwen2-1.5B-Instruct",
                 max_text_length: int = 512,
                 processor=None,
                 image_size: int = 448):
        """
        Args:
            data: Path to JSON file, list of OneSample, or list of dicts
            image_dir: Directory containing images
            tokenizer_name: HuggingFace tokenizer
            max_text_length: Max length for text tokenization
            processor: Image processor (VinternProcessor or similar)
            image_size: Image size
        """
        self.image_dir = Path(image_dir)
        self.max_text_length = max_text_length
        self.image_size = image_size
        
        # Load data
        if isinstance(data, str):
            data_loader_logger.info(f"Loading fine-tune dataset from JSON: {data}")
            self.data = load_dataset_from_json(data, str(self.image_dir))
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], OneSample):
                data_loader_logger.info(f"Using {len(data)} OneSample objects")
                self.data = data
            else:
                # Auto-convert dicts
                data_loader_logger.info(f"Converting {len(data)} dicts to OneSample objects")
                self.data = self._convert_dicts_to_samples(data)
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Processor (if provided)
        self.processor = processor
        
        data_loader_logger.info(f"FineTuneDataset initialized with {len(self.data)} samples")
    
    @staticmethod
    def _convert_dicts_to_samples(data_list: List[Dict]) -> List[OneSample]:
        """Convert dict format to OneSample objects."""
        samples = []
        for idx, item in enumerate(data_list):
            try:
                image_path = item.get('image_path') or os.path.join('.', item.get('image', ''))
                if not os.path.exists(image_path):
                    continue
                
                sample = OneSample(
                    image_path=image_path,
                    question=item.get('question', ''),
                    answer=item.get('answer', ''),
                    metadata=item.get('metadata', {})
                )
                samples.append(sample)
            except Exception as e:
                data_loader_logger.warning(f"Error converting item {idx}: {e}")
                continue
        return samples
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Load image
        try:
            image = Image.open(item.image_path).convert('RGB')
        except Exception as e:
            data_loader_logger.warning(f"Failed to load image {item.image_path}: {e}")
            # Return placeholder and skip this sample
            return self.__getitem__((idx + 1) % len(self.data))
        
        # Process image if processor provided
        if self.processor is not None:
            image_data = self.processor.preprocess(image)
            pixel_values = image_data['pixel_values']
        else:
            # Simple resize + normalize if no processor
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image)
        
        # Format text prompt
        # For QA: "Question: {question} Answer: {answer}"
        text = f"Question: {item.question}\nAnswer: {item.answer}"
        
        # Tokenize
        text_encoding = self.tokenizer(
            text,
            max_length=self.max_text_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'image_path': item.image_path,
            'pixel_values': pixel_values,
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'text': text,
        }
