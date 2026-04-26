"""
Unified dataset loader supporting both local and Kaggle environments.
Handles raw data loading and conversion to OneSample format.
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Union
from tqdm import tqdm

from src.schema.data_schema import OneSample
from src.middleware.logger import data_loader_logger
from .data_actions import get_all_image_paths
from .environment import DataPathResolver, EnvironmentDetector


class UnifiedDataLoader:
    """Load data from local or Kaggle environments with unified OneSample format."""
    
    def __init__(self, data_config: dict, kaggle_config: dict, project_root: str = "."):
        """
        Initialize unified data loader.
        
        Args:
            data_config: Configuration dict with 'data_path' keys
            kaggle_config: Configuration dict with 'kaggle_setup' keys
            project_root: Root project directory
        """
        self.resolver = DataPathResolver(data_config, kaggle_config, project_root)
        self.is_kaggle = self.resolver.is_kaggle
        
        env_name = "Kaggle" if self.is_kaggle else "Local"
        data_loader_logger.info(f"Initialized UnifiedDataLoader in {env_name} environment")
    
    def load_raw_data(self, max_samples: Optional[int] = None) -> List[OneSample]:
        """
        Load raw data from images and text CSV.
        
        Args:
            max_samples: Maximum number of samples to load (default: None = all)
        
        Returns:
            List of OneSample objects
        """
        # Validate paths exist
        self.resolver.validate_raw_data_exists()
        
        # Get paths
        images_dir = self.resolver.get_raw_images_dir()
        texts_file = self.resolver.get_raw_texts_file()
        
        data_loader_logger.info(f"Loading raw data from:")
        data_loader_logger.info(f"  Images: {images_dir}")
        data_loader_logger.info(f"  Text CSV: {texts_file}")
        
        # Load text data
        df = pd.read_csv(str(texts_file))
        
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            data_loader_logger.info(f"Limiting to {max_samples} samples")
        
        data_loader_logger.info(f"Total samples: {len(df)}")
        
        # Get all image paths and create mapping by basename
        image_paths = get_all_image_paths(str(images_dir))
        image_path_map = {Path(p).name: p for p in image_paths}
        
        data_loader_logger.info(f"Found {len(image_path_map)} images")
        
        # Load samples
        samples = []
        
        # Determine image ID column
        image_col = self._find_image_column(df)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading samples"):
            try:
                image_id = row[image_col]
                image_path = None
                
                # Try different image naming formats
                if isinstance(image_id, str):
                    # String format - could be filename or numeric string
                    if image_path_map.get(image_id):
                        image_path = image_path_map[image_id]
                    elif image_path_map.get(image_id + '.jpg'):
                        image_path = image_path_map[image_id + '.jpg']
                    # Try extracting from URL if it has path separators
                    elif '/' in image_id:
                        filename = image_id.split('/')[-1]
                        image_path = image_path_map.get(filename)
                else:
                    # Numeric format - convert to 12-digit padded filename
                    padded_name = f"{int(image_id):012d}.jpg"
                    image_path = image_path_map.get(padded_name)
                    
                    # Also try without padding or with variations
                    if not image_path:
                        str_id = str(int(image_id))
                        image_path = image_path_map.get(str_id + '.jpg')
                
                if not image_path:
                    # Last attempt: extract from image_url if available
                    if 'image_url' in df.columns and pd.notna(row.get('image_url')):
                        url = str(row['image_url'])
                        filename = url.split('/')[-1]
                        image_path = image_path_map.get(filename)
                
                if not image_path:
                    data_loader_logger.debug(f"Image not found for {image_col}={image_id}")
                    continue
                
                # Extract question and answers
                question = str(row.get('question', ''))
                
                # Handle answers - should be list of answers (like ref1/)
                answers_val = row.get('answers', row.get('answer', []))
                answers_list = []
                
                if pd.isna(answers_val):
                    answers_list = ['']
                elif isinstance(answers_val, str):
                    try:
                        # Try parsing as list representation string: "['answer1', 'answer2']"
                        import ast
                        answers_list = ast.literal_eval(answers_val)
                        if not isinstance(answers_list, list):
                            answers_list = [str(answers_list)]
                    except:
                        # Fallback: treat as single answer
                        answers_list = [str(answers_val)]
                elif isinstance(answers_val, list):
                    # Already a list
                    answers_list = [str(a).strip() for a in answers_val]
                else:
                    # Single value
                    answers_list = [str(answers_val)]
                
                # Filter out empty strings and strip whitespace
                answers_list = [a.strip() for a in answers_list if a and str(a).strip()]
                if not answers_list:
                    answers_list = ['']
                
                sample = OneSample(
                    image_path=image_path,
                    question=question.strip(),
                    answers=answers_list
                )
                samples.append(sample)
                
            except Exception as e:
                data_loader_logger.debug(f"Error loading sample {idx}: {e}")
                continue
        
        data_loader_logger.info(f"Successfully loaded {len(samples)} samples")
        return samples

    
    def _find_image_column(self, df: pd.DataFrame) -> str:
        """Detect image ID column name from available columns."""
        possible_names = ['image_id', 'image_name', 'image_link', 'image_url', 'image_file']
        
        for col in possible_names:
            if col in df.columns:
                return col
        
        # Fallback: find first column that looks like an image reference
        for col in df.columns:
            if 'image' in col.lower():
                return col
        
        raise ValueError(
            f"Could not find image column. Available columns: {df.columns.tolist()}"
        )
    
    def load_from_json(self, json_path: str) -> List[OneSample]:
        """
        Load OneSample objects from JSON file.
        
        Args:
            json_path: Path to JSON file containing OneSample data
        
        Returns:
            List of OneSample objects
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            samples.append(OneSample(**item))
        
        data_loader_logger.info(f"Loaded {len(samples)} samples from {json_path}")
        return samples
    
    def save_to_json(self, samples: List[OneSample], output_path: str) -> None:
        """Save OneSample objects to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = [sample.to_dict() for sample in samples]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        data_loader_logger.info(f"Saved {len(samples)} samples to {output_path}")
