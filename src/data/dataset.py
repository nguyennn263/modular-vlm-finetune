"""
VLM Dataset cho training Vision-Language Model
Aligned with ref project's dataset structure for consistency
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, List, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .processor import VinternProcessor
from .data_actions import load_dataset_from_json
from src.schema.data_schema import OneSample
from src.middleware.logger import data_loader_logger


class VLMDataset(Dataset):
    """Dataset cho Vision-Language Model training - supports both dict and OneSample formats"""
    
    def __init__(
        self,
        data: Union[str, List[OneSample], List[Dict]],
        image_dir: str,
        tokenizer_name: str = "Qwen/Qwen2-1.5B-Instruct",
        max_length: int = 2048,
        image_size: int = 448,
        max_tiles: int = 12,
        image_token: str = "<image>",
    ):
        """
        Args:
            data: Either path to JSON file, or list of OneSample objects, or list of dicts
            image_dir: Directory containing images (required if data is file path or dicts)
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            image_size: Image size for processor
            max_tiles: Maximum number of tiles for image processing
            image_token: Image token for prompt
        """
        self.image_dir = Path(image_dir)
        self.max_length = max_length
        self.image_token = image_token
        
        # Load data - support multiple formats
        if isinstance(data, str):
            # Load from JSON file
            data_loader_logger.info(f"Loading dataset from JSON: {data}")
            self.data = load_dataset_from_json(data, str(self.image_dir))
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], OneSample):
                # Already OneSample objects
                data_loader_logger.info(f"Using {len(data)} OneSample objects")
                self.data = data
            else:
                # Convert dicts to OneSample objects
                data_loader_logger.info(f"Converting {len(data)} dicts to OneSample objects")
                self.data = self._convert_dicts_to_samples(data, str(self.image_dir))
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
        
        # Tokenizer
        data_loader_logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            trust_remote_code=True
        )
        
        # Image processor
        self.processor = VinternProcessor(
            image_size=image_size,
            max_tiles=max_tiles,
        )
        
        # Prompt template - SIMPLIFIED: output format only
        # Format: <image>\n{question}\nAnswer: [model generates this]
        # Note: question from data already has <image>\n at the start
        # Loss computed ONLY on Answer tokens
        self.answer_header = "Answer: "
        
        data_loader_logger.info(f"VLMDataset initialized with {len(self.data)} samples")
    
    @staticmethod
    def _convert_dicts_to_samples(data_list: List[Dict], image_dir: str) -> List[OneSample]:
        """Convert dictionary format to OneSample objects"""
        samples = []
        for idx, item in enumerate(data_list):
            try:
                # Extract image filename
                image_filename = item.get('image', '')
                if not image_filename:
                    data_loader_logger.warning(f"Item {idx}: missing 'image' field, skipping")
                    continue
                
                # Get the actual image path
                image_filename_only = os.path.basename(image_filename)
                potential_path = os.path.join(image_dir, image_filename_only)
                
                if os.path.exists(potential_path):
                    image_path = potential_path
                else:
                    data_loader_logger.warning(f"Item {idx}: Image not found - {image_filename}")
                    continue
                
                # Extract question and answer
                question = item.get('question', '')
                answer = item.get('answer', '')
                
                if not question or not answer:
                    data_loader_logger.warning(f"Item {idx}: missing question or answer, skipping")
                    continue
                
                # Create OneSample
                sample = OneSample(
                    image_path=image_path,
                    question=question,
                    answer=answer,
                    metadata={"original_filename": image_filename_only}
                )
                samples.append(sample)
            except Exception as e:
                data_loader_logger.error(f"Error converting item {idx}: {e}")
                continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Handle both OneSample objects and dicts
        if isinstance(item, OneSample):
            image_path = item.image_path
            question = item.question
            answer = item.answer
        else:
            # Fallback for dict format
            if os.path.isabs(item.get("image", "")):
                image_path = item["image"]
            else:
                image_path = os.path.join(str(self.image_dir), item["image"])
            question = item["question"]
            answer = item["answer"]
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            data_loader_logger.warning(f"Failed to load image {image_path}: {e}, using black placeholder")
            image = Image.new('RGB', (448, 448), (0, 0, 0))
        
        # Process image
        image_data = self.processor.preprocess(image)
        
        # Format text with correct format matching notebook
        # Question from data already has: "<image>\n{actual question}"
        # So just append answer part
        full_text = f"{question}\n{self.answer_header}{answer}"
        
        # Tokenize FULL text including answer
        # This will be used for loss computation (decoder mode)
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        
        # Now tokenize WITHOUT answer to find where answer begins
        # This is: "<image>\n{question}\nAnswer: "
        question_part = f"{question}\n{self.answer_header}"
        question_encoding = self.tokenizer(
            question_part,
            add_special_tokens=False,  # Don't add <bos> etc
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        
        # answer_start_pos: where answer begins in token stream
        # Loss will only be computed from this position onwards
        answer_start_pos = question_encoding["input_ids"].shape[1]
        
        return {
            "input_ids": full_encoding["input_ids"].squeeze(0),
            "attention_mask": full_encoding["attention_mask"].squeeze(0),
            "pixel_values": image_data["pixel_values"],
            "num_patches": image_data["num_patches"],
            "question": question,
            "answer": answer,
            "answer_start_pos": answer_start_pos,  # NEW: for loss masking
        }



class ConversationDataset(Dataset):
    """Dataset cho multi-turn conversation với images - supports both dict and OneSample formats"""
    
    def __init__(
        self,
        data: Union[str, List[Dict]],
        image_dir: str,
        tokenizer_name: str = "Qwen/Qwen2-1.5B-Instruct",
        max_length: int = 2048,
        image_size: int = 448,
        max_tiles: int = 12,
    ):
        """
        Args:
            data: Either path to JSON file or list of conversation dicts
            image_dir: Directory containing images
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            image_size: Image size for processor
            max_tiles: Maximum number of tiles for image processing
        """
        self.image_dir = Path(image_dir)
        self.max_length = max_length
        
        # Load data
        if isinstance(data, str):
            data_loader_logger.info(f"Loading conversation dataset from JSON: {data}")
            with open(data) as f:
                self.data = json.load(f)
        elif isinstance(data, list):
            data_loader_logger.info(f"Using {len(data)} conversation items")
            self.data = data
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        self.processor = VinternProcessor(
            image_size=image_size,
            max_tiles=max_tiles,
        )
        
        data_loader_logger.info(f"ConversationDataset initialized with {len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def format_conversation(self, conversations: List[Dict]) -> str:
        """Format multi-turn conversation theo Qwen2 format"""
        text = "<|im_start|>system\nYou are a helpful AI assistant that can analyze images.<|im_end|>\n"
        
        for turn in conversations:
            # Handle both 'role' and 'from' field names
            role = turn.get("role") or turn.get("from", "user")
            # Map role names: "human" -> "user", "gpt" -> "assistant"
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            
            content = turn.get("content") or turn.get("value", "")
            
            # Insert <image> token in first user message if not present
            if role == "user" and "<image>" not in content and "image" in str(turn):
                content = "<image>\n" + content
            
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        return text
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Load image if exists
        pixel_values = None
        num_patches = 0
        
        if "image" in item:
            try:
                image_filename = item["image"]
                if os.path.isabs(image_filename):
                    image_path = image_filename
                else:
                    image_path = os.path.join(str(self.image_dir), image_filename)
                
                image = Image.open(image_path).convert("RGB")
                image_data = self.processor.preprocess(image)
                pixel_values = image_data["pixel_values"]
                num_patches = image_data["num_patches"]
            except Exception as e:
                data_loader_logger.warning(f"Failed to load image at idx {idx}: {e}, using black placeholder")
                # Create black placeholder image and process it
                image = Image.new('RGB', (448, 448), (0, 0, 0))
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
