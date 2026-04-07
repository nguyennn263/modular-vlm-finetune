"""
Memory-efficient lazy-loading data loaders for bridge fine-tuning.
Loads images and text on-the-fly instead of pre-loading everything into RAM.
"""

import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import numpy as np
from tqdm import tqdm


class LazyVQADataset(Dataset):
    """
    Lazy-loading VQA dataset that loads images on-the-fly.
    Reduces memory footprint from ~70GB to ~100MB.
    """
    
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        img_size: int = 448,
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        """
        Initialize lazy dataset.
        
        Args:
            csv_path: Path to CSV with columns: image_id, question, answer
            images_dir: Directory containing images
            img_size: Image size for resizing
            max_samples: Limit number of samples (default: None = all)
            seed: Random seed
        """
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.seed = seed
        
        # Load only metadata (lazy)
        print(f"Loading metadata from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        if max_samples and len(self.df) > max_samples:
            self.df = self.df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
            print(f"Limiting to {max_samples} samples")
        
        print(f"Total samples: {len(self.df)}")
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image on-demand
        try:
            # Extract image filename
            if 'image_name' in self.df.columns and pd.notna(row.get('image_name')):
                img_name = str(row['image_name'])
            elif 'image_url' in self.df.columns and pd.notna(row.get('image_url')):
                url = str(row['image_url'])
                img_name = url.split('/')[-1]
            else:
                # Fallback: use black image
                pixel_values = torch.zeros(3, self.img_size, self.img_size)
                question = str(row.get('question', ''))
                tokens, mask = self._tokenize_question(question)
                return pixel_values, tokens, mask
            
            # Add .jpg if missing
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_name = img_name + '.jpg'
            
            img_path = self.images_dir / img_name
            
            # Try alternative extensions
            if not img_path.exists():
                for ext in ['.jpg', '.jpeg', '.png', '.PNG', '.JPG']:
                    alt_path = img_path.parent / (img_path.stem + ext)
                    if alt_path.exists():
                        img_path = alt_path
                        break
            
            # Load image
            if img_path.exists():
                image = Image.open(img_path).convert('RGB')
                pixel_values = self.transform(image)
            else:
                # Fallback to black image
                pixel_values = torch.zeros(3, self.img_size, self.img_size)
        
        except Exception as e:
            print(f"Warning: Failed to load image at {img_path}: {e}")
            pixel_values = torch.zeros(3, self.img_size, self.img_size)
        
        # Tokenize question
        question = str(row.get('question', ''))
        input_ids, attention_mask = self._tokenize_question(question)
        
        return pixel_values, input_ids, attention_mask
    
    def _tokenize_question(self, question: str, max_len: int = 50):
        """Simple tokenization: split into tokens and encode as char codes"""
        tokens = question.split()[:max_len]
        token_ids = [ord(c) % 10000 for word in tokens for c in word][:max_len]
        
        # Pad to max_len
        while len(token_ids) < max_len:
            token_ids.append(0)
        token_ids = token_ids[:max_len]
        
        return torch.tensor(token_ids, dtype=torch.long), torch.ones(max_len, dtype=torch.long)


def load_lazy_datasets(
    csv_path: str,
    images_dir: str,
    val_ratio: float = 0.1,
    img_size: int = 448,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Load datasets with lazy image loading (memory-efficient).
    
    Args:
        csv_path: Path to CSV
        images_dir: Directory with images
        val_ratio: Validation split (default 0.1)
        img_size: Image size (default 448)
        max_samples: Limit samples (default None = all)
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Create full dataset
    full_dataset = LazyVQADataset(
        csv_path=csv_path,
        images_dir=images_dir,
        img_size=img_size,
        max_samples=max_samples,
        seed=seed
    )
    
    # Split
    n_samples = len(full_dataset)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val
    
    torch.manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_dataset, val_dataset
