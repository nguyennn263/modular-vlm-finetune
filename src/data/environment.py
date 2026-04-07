"""
Environment detection and path resolution for local and Kaggle environments.
"""
import os
from pathlib import Path
from typing import Optional


class EnvironmentDetector:
    """Detects runtime environment (local vs Kaggle) and resolves data paths accordingly."""
    
    KAGGLE_INPUT_PATH = Path("/kaggle/input")
    
    @classmethod
    def is_kaggle(cls) -> bool:
        """Check if code is running on Kaggle."""
        return cls.KAGGLE_INPUT_PATH.exists()
    
    @classmethod
    def get_kaggle_project_path(cls, kaggle_project: str) -> Optional[Path]:
        """Get absolute path to Kaggle project in /kaggle/input/."""
        if not cls.is_kaggle():
            return None
        
        # Kaggle datasets are named by their slug (e.g., 'nguynrichard/auto-vqabest' -> 'auto-vqabest')
        project_name = kaggle_project.split('/')[-1]
        project_path = cls.KAGGLE_INPUT_PATH / project_name
        
        if not project_path.exists():
            raise FileNotFoundError(
                f"Kaggle project '{project_name}' not found at {project_path}\n"
                f"Available datasets in /kaggle/input/:\n"
                f"{list(cls.KAGGLE_INPUT_PATH.iterdir())}"
            )
        
        return project_path


class DataPathResolver:
    """Resolves data paths based on environment and configuration."""
    
    def __init__(self, data_config: dict, kaggle_config: dict, project_root: str = "."):
        """
        Initialize path resolver.
        
        Args:
            data_config: dict with 'data_path' containing raw_images, raw_texts, etc.
            kaggle_config: dict with 'kaggle_setup' containing kaggle_project, images_folder, text_file
            project_root: Project root directory (default: current directory)
        """
        self.data_config = data_config
        self.kaggle_config = kaggle_config
        self.project_root = Path(project_root)
        self.is_kaggle = EnvironmentDetector.is_kaggle()
    
    def get_raw_images_dir(self) -> Path:
        """Get raw images directory path."""
        if self.is_kaggle:
            kaggle_project = self.kaggle_config['kaggle_project']
            project_path = EnvironmentDetector.get_kaggle_project_path(kaggle_project)
            images_folder = self.kaggle_config['images_folder']
            return project_path / images_folder
        else:
            return self.project_root / self.data_config['data_path']['raw_images']
    
    def get_raw_texts_file(self) -> Path:
        """Get raw texts CSV file path."""
        if self.is_kaggle:
            kaggle_project = self.kaggle_config['kaggle_project']
            project_path = EnvironmentDetector.get_kaggle_project_path(kaggle_project)
            text_file = self.kaggle_config['text_file']
            return project_path / text_file
        else:
            # Construct path to text file in local raw_texts folder
            raw_texts = self.project_root / self.data_config['data_path']['raw_texts']
            # Assume text file is the CSV in raw_texts folder
            csv_files = list(raw_texts.glob('*.csv'))
            if not csv_files:
                raise FileNotFoundError(f"No CSV file found in {raw_texts}")
            return csv_files[0]
    
    def get_processed_train_images_dir(self) -> Path:
        """Get processed training images directory."""
        return self.project_root / self.data_config['data_path']['processed_data']['train_data']['images_folder']
    
    def get_processed_train_texts_dir(self) -> Path:
        """Get processed training texts directory."""
        return self.project_root / self.data_config['data_path']['processed_data']['train_data']['texts_folder']
    
    def get_processed_val_images_dir(self) -> Path:
        """Get processed validation images directory."""
        return self.project_root / self.data_config['data_path']['processed_data']['val_data']['images_folder']
    
    def get_processed_val_texts_dir(self) -> Path:
        """Get processed validation texts directory."""
        return self.project_root / self.data_config['data_path']['processed_data']['val_data']['texts_folder']
    
    def validate_raw_data_exists(self) -> bool:
        """Validate that raw data paths exist."""
        images_dir = self.get_raw_images_dir()
        texts_file = self.get_raw_texts_file()
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not texts_file.exists():
            raise FileNotFoundError(f"Text file not found: {texts_file}")
        
        return True
