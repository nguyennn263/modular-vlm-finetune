"""
Data action utilities for loading and processing datasets.
Mirrors ref project's data_actions.py for consistency.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Union
from tqdm import tqdm

from src.schema.data_schema import OneSample
from src.middleware.logger import data_loader_logger


def load_image_path(image_path: str) -> str:
    """Validate and return image path.

    Args:
        image_path: Path to the image file.
    
    Returns:
        Valid path to the image.
    
    Raises:
        FileNotFoundError: If image file does not exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image_path


def get_all_image_paths(directory: str) -> list:
    """Get a list of all image file paths in the specified directory.

    Args:
        directory: Path to the directory containing images.
    
    Returns:
        List of image file paths.
    """
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    data_loader_logger.info(f"Found {len(image_paths)} images in directory: {directory}")
    return image_paths


def load_json_data(json_path: str) -> Union[List, Dict]:
    """Load data from JSON file.

    Args:
        json_path: Path to the JSON file.
    
    Returns:
        Loaded JSON data (list or dict).
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data_loader_logger.info(f"Loaded {len(data)} items from {json_path}")
        return data
    except Exception as e:
        data_loader_logger.error(f"Failed to load JSON from {json_path}: {e}")
        raise


def load_dataset_from_json(
    json_path: str, 
    image_dir: str
) -> List[OneSample]:
    """Load dataset from JSON file and match with images.

    Args:
        json_path: Path to the JSON file containing data items.
        image_dir: Directory containing images.
    
    Returns:
        List of OneSample instances.
    """
    data_loader_logger.info(f"Loading dataset from JSON: {json_path}")
    
    # Load JSON data
    data_list = load_json_data(json_path)
    
    # Ensure data is a list
    if not isinstance(data_list, list):
        data_list = [data_list]
    
    # Get all image paths
    data_loader_logger.info(f"Retrieving image paths from directory: {image_dir}")
    image_paths = get_all_image_paths(image_dir)
    
    # Create a mapping from filename to full path
    image_path_map = {os.path.basename(p): p for p in image_paths}
    
    # Process each item
    data_samples = []
    for idx, item in tqdm(enumerate(data_list), total=len(data_list), desc="Loading data samples"):
        try:
            # Extract image filename (support both relative and absolute paths)
            image_filename = item.get('image', '')
            if not image_filename:
                data_loader_logger.warning(f"Item {idx}: missing 'image' field, skipping")
                continue
            
            # Get the actual image path
            image_filename_only = os.path.basename(image_filename)
            if image_filename_only in image_path_map:
                image_path = image_path_map[image_filename_only]
            elif os.path.isabs(image_filename) and os.path.exists(image_filename):
                image_path = image_filename
            else:
                # Try as relative path
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
            data_samples.append(sample)
        
        except Exception as e:
            data_loader_logger.error(f"Error processing item {idx}: {e}")
            continue
    
    data_loader_logger.info(f"Successfully loaded {len(data_samples)} out of {len(data_list)} samples")
    return data_samples


def load_train_val_datasets(
    data_dir: str,
    image_dir: str,
    train_file: str = "train.json",
    val_file: str = "val.json"
) -> tuple:
    """Load both training and validation datasets.

    Args:
        data_dir: Directory containing JSON files.
        image_dir: Directory containing images.
        train_file: Name of training JSON file.
        val_file: Name of validation JSON file.
    
    Returns:
        Tuple of (train_samples, val_samples).
    """
    train_path = os.path.join(data_dir, train_file)
    val_path = os.path.join(data_dir, val_file)
    
    train_samples = load_dataset_from_json(train_path, image_dir) if os.path.exists(train_path) else []
    val_samples = load_dataset_from_json(val_path, image_dir) if os.path.exists(val_path) else []
    
    return train_samples, val_samples
