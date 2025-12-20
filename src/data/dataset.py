"""
VLM Dataset cho training Vision-Language Model
"""
import json
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .processor import VinternProcessor


class VLMDataset(Dataset):
    """Dataset cho Vision-Language Model training"""
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer_name: str = "Qwen/Qwen2-1.5B-Instruct",
        max_length: int = 2048,
        image_size: int = 448,
        max_tiles: int = 12,
        image_token: str = "<image>",
    ):
        self.image_dir = Path(image_dir)
        self.max_length = max_length
        self.image_token = image_token
        
        # Load data
        with open(data_path) as f:
            self.data = json.load(f)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            trust_remote_code=True
        )
        
        # Image processor
        self.processor = VinternProcessor(
            image_size=image_size,
            max_tiles=max_tiles,
        )
        
        # Prompt template
        self.prompt_template = (
            "<|im_start|>system\n"
            "Bạn là một trợ lý AI thông minh, có khả năng phân tích hình ảnh.<|im_end|>\n"
            "<|im_start|>user\n"
            "<image>\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{answer}<|im_end|>"
        )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Load image
        image_path = self.image_dir / item["image"]
        image = Image.open(image_path).convert("RGB")
        
        # Process image
        image_data = self.processor.preprocess(image)
        
        # Format text
        text = self.prompt_template.format(
            question=item["question"],
            answer=item["answer"],
        )
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "pixel_values": image_data["pixel_values"],
            "num_patches": image_data["num_patches"],
            "question": item["question"],
            "answer": item["answer"],
        }


class ConversationDataset(Dataset):
    """Dataset cho multi-turn conversation với images"""
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer_name: str = "Qwen/Qwen2-1.5B-Instruct",
        max_length: int = 2048,
        image_size: int = 448,
        max_tiles: int = 12,
    ):
        self.image_dir = Path(image_dir)
        self.max_length = max_length
        
        with open(data_path) as f:
            self.data = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        self.processor = VinternProcessor(
            image_size=image_size,
            max_tiles=max_tiles,
        )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def format_conversation(self, conversations: List[Dict]) -> str:
        """Format multi-turn conversation"""
        text = "<|im_start|>system\nBạn là trợ lý AI.<|im_end|>\n"
        
        for turn in conversations:
            role = turn["role"]
            content = turn["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        return text
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Load image if exists
        pixel_values = None
        num_patches = 0
        
        if "image" in item:
            image_path = self.image_dir / item["image"]
            image = Image.open(image_path).convert("RGB")
            image_data = self.processor.preprocess(image)
            pixel_values = image_data["pixel_values"]
            num_patches = image_data["num_patches"]
        
        # Format conversation
        text = self.format_conversation(item["conversations"])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        
        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
            result["num_patches"] = num_patches
        
        return result
