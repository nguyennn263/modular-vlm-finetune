"""
Data loaders for bridge fine-tuning experiments.
Loads real images and text data from configured paths.
"""

import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import TensorDataset
from typing import Tuple, Optional, Union
import numpy as np
from tqdm import tqdm


def load_datasets(
    csv_path: str,
    images_dir: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
    img_size: int = 448,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> Tuple[TensorDataset, TensorDataset, Optional[TensorDataset]]:
    """
    Load real image and text data from CSV and image directory.
    
    Args:
        csv_path: Path to CSV file with columns: image_id, question, answer
        images_dir: Directory containing images
        val_ratio: Validation split ratio (default: 0.1)
        test_ratio: Test split ratio (default: 0.0 = no test split)
        img_size: Image size for processing (default: 448)
        max_samples: Limit number of samples (default: None = use all)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        If test_ratio=0, test_dataset is None
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
        print(f"Limiting to {max_samples} samples")
    
    print(f"Total samples: {len(df)}")
    
    # Prepare image transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    pixel_values = []
    input_ids = []
    attention_masks = []
    
    images_dir = Path(images_dir)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading data"):
        # Load image
        try:
            # Extract image filename from available columns
            if 'image_name' in df.columns and pd.notna(row.get('image_name')):
                img_name = str(row['image_name'])
            elif 'image_url' in df.columns and pd.notna(row.get('image_url')):
                # Extract filename from URL: http://.../.../000000581569.jpg -> 000000581569.jpg
                url = str(row['image_url'])
                img_name = url.split('/')[-1]
            else:
                # Use black placeholder if no image name found
                pixel_values.append(torch.zeros(3, img_size, img_size))
                question = str(row.get('question', ''))
                tokens = question.split()[:50]
                token_ids = [ord(c) % 10000 for word in tokens for c in word][:50]
                while len(token_ids) < 50:
                    token_ids.append(0)
                token_ids = token_ids[:50]
                input_ids.append(torch.tensor(token_ids))
                attention_masks.append(torch.ones(50))
                continue
            
            # Add .jpg if missing
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_name = img_name + '.jpg'
            
            img_path = images_dir / img_name
            
            if not img_path.exists():
                # Try with common extensions
                for ext in ['.jpg', '.jpeg', '.png', '.PNG', '.JPG']:
                    alt_path = img_path.parent / (img_path.stem + ext)
                    if alt_path.exists():
                        img_path = alt_path
                        break
            
            image = Image.open(img_path).convert('RGB')
            image = transform(image)
            pixel_values.append(image)
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            # Use black placeholder
            pixel_values.append(torch.zeros(3, img_size, img_size))
        
        # Tokenize question (simple: split into tokens and pad)
        question = str(row.get('question', ''))
        tokens = question.split()[:50]  # Max 50 tokens
        token_ids = [ord(c) % 10000 for word in tokens for c in word][:50]
        
        # Pad to 50 tokens
        while len(token_ids) < 50:
            token_ids.append(0)
        token_ids = token_ids[:50]
        
        input_ids.append(torch.tensor(token_ids, dtype=torch.long))
        attention_masks.append(torch.ones(50, dtype=torch.long))
    
    # Stack tensors
    pixel_values = torch.stack(pixel_values)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    
    # Split train/val/test
    n_samples = len(df)
    n_val = int(n_samples * val_ratio)
    n_test = int(n_samples * test_ratio)
    n_train = n_samples - n_val - n_test
    
    indices = torch.randperm(n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:] if n_test > 0 else None
    
    train_dataset = TensorDataset(
        pixel_values[train_indices],
        input_ids[train_indices],
        attention_masks[train_indices]
    )
    
    val_dataset = TensorDataset(
        pixel_values[val_indices],
        input_ids[val_indices],
        attention_masks[val_indices]
    )
    
    test_dataset = None
    if test_indices is not None and len(test_indices) > 0:
        test_dataset = TensorDataset(
            pixel_values[test_indices],
            input_ids[test_indices],
            attention_masks[test_indices]
        )
    
    info = f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}"
    if test_dataset is not None:
        info += f", Test samples: {len(test_dataset)}"
    print(info)
    
    if test_dataset is not None:
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, val_dataset, None
