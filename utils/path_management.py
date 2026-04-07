from pathlib import Path

# Root directory
ROOT_DIR = Path(__file__).resolve().parents[1]

# Legacy paths (for backward compatibility - assumes local environment)
RAW_TEXT_CSV = ROOT_DIR / 'data' / 'raw' / 'texts' / 'evaluate_60k_data_balanced_preprocessed.csv'
RAW_IMAGES_DIR = ROOT_DIR / 'data' / 'raw' / 'images'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'


def get_raw_data_paths():
    """
    Get raw data paths for current environment (local or Kaggle).
    
    Returns:
        Tuple of (images_dir, texts_file) Path objects
    """
    from src.data.environment import DataPathResolver
    from utils.config_loader import load_config
    
    data_config = load_config(str(ROOT_DIR / 'configs/data_configs.yaml'))
    
    resolver = DataPathResolver(
        data_config,
        data_config['kaggle_setup'],
        str(ROOT_DIR)
    )
    
    return resolver.get_raw_images_dir(), resolver.get_raw_texts_file()
