"""
Device detection and auto-optimization for training.
Automatically selects optimal training parameters based on available GPU memory.
"""

import torch
import logging
from pathlib import Path
import yaml
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DeviceDetector:
    """Detect GPU capabilities and select optimal training profile."""
    
    def __init__(self, config_path: str = "configs/device_profiles.yaml"):
        """Initialize device detector with profiles."""
        self.config_path = Path(config_path)
        self.profiles = self._load_profiles()
        self.device_info = self._detect_device()
        self.selected_profile = self._select_profile()
    
    def _load_profiles(self) -> Dict:
        """Load device profiles from config file."""
        if not self.config_path.exists():
            logger.warning(f"Device profiles not found at {self.config_path}")
            return {}
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config.get('devices', {})
    
    def _detect_device(self) -> Dict:
        """Detect GPU and memory information."""
        device_info = {
            'device': 'cpu',
            'name': 'CPU',
            'memory_gb': 0,
            'cuda_available': False,
            'gpu_name': 'None',
        }
        
        if torch.cuda.is_available():
            device_info['device'] = 'cuda'
            device_info['cuda_available'] = True
            device_info['gpu_name'] = torch.cuda.get_device_name(0)
            
            # Get total GPU memory in GB
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024 ** 3)
            device_info['memory_gb'] = round(memory_gb, 1)
        
        return device_info
    
    def _select_profile(self) -> Optional[Dict]:
        """Select optimal profile based on detected hardware."""
        if not self.device_info['cuda_available']:
            logger.warning("CUDA not available - CPU training will be very slow")
            return None
        
        memory_gb = self.device_info['memory_gb']
        
        # Find matching profile
        for profile_key, profile_config in self.profiles.items():
            if (profile_config['min_memory_gb'] <= memory_gb <= 
                profile_config['max_memory_gb']):
                
                profile_config['name'] = profile_key
                logger.info(f"✓ Auto-selected profile: {profile_config['name']}")
                logger.info(f"  Description: {profile_config['description']}")
                logger.info(f"  GPU: {self.device_info['gpu_name']}")
                logger.info(f"  Memory: {memory_gb} GB")
                logger.info(f"  Batch size: {profile_config['batch_size']}")
                logger.info(f"  Gradient accumulation: {profile_config['gradient_accumulation_steps']}")
                logger.info(f"  Seq length: {profile_config['max_seq_length']}")
                
                return profile_config
        
        # Fallback: use largest profile if memory exceeds all
        if memory_gb > 40:
            logger.info("✓ Memory exceeds 40GB - using L40 profile")
            profile = self.profiles.get('l40_45gb', {})
            profile['name'] = 'l40_45gb'
            return profile
        
        # Fallback: use smallest profile if memory below all
        logger.warning(f"Memory {memory_gb}GB doesn't match any profile, using T4 (minimal)")
        profile = self.profiles.get('t4_16gb', {})
        profile['name'] = 't4_16gb'
        return profile
    
    def get_training_config(self) -> Dict:
        """Get training configuration for current device."""
        if not self.selected_profile:
            raise RuntimeError("No suitable device profile found")
        
        return {
            'batch_size': self.selected_profile.get('batch_size', 8),
            'gradient_accumulation_steps': self.selected_profile.get('gradient_accumulation_steps', 1),
            'max_seq_length': self.selected_profile.get('max_seq_length', 256),
            'num_workers': self.selected_profile.get('num_workers', 4),
            'eval_steps': self.selected_profile.get('eval_steps', 100),
            'save_steps': self.selected_profile.get('save_steps', 500),
            'use_flash_attn': self.selected_profile.get('use_flash_attn', False),
            'mixed_precision': self.selected_profile.get('mixed_precision', 'bf16'),
        }
    
    def print_info(self):
        """Print device and configuration info."""
        logger.info("\n" + "="*60)
        logger.info("DEVICE DETECTION & AUTO-OPTIMIZATION")
        logger.info("="*60)
        logger.info(f"GPU: {self.device_info['gpu_name']}")
        logger.info(f"Memory: {self.device_info['memory_gb']} GB")
        logger.info(f"CUDA Available: {self.device_info['cuda_available']}")
        
        if self.selected_profile:
            logger.info(f"\nSelected Profile: {self.selected_profile['name']}")
            config = self.get_training_config()
            logger.info(f"Batch Size: {config['batch_size']}")
            logger.info(f"Gradient Accumulation: {config['gradient_accumulation_steps']}")
            logger.info(f"Max Seq Length: {config['max_seq_length']}")
            logger.info(f"Num Workers: {config['num_workers']}")
            logger.info(f"Eval Steps: {config['eval_steps']}")
            logger.info(f"Use Flash Attention: {config['use_flash_attn']}")
        
        logger.info("="*60 + "\n")


# Convenience functions
def auto_configure_training() -> Dict:
    """Auto-detect device and return optimal training config."""
    detector = DeviceDetector()
    detector.print_info()
    return detector.get_training_config()


def get_device_profile(profile_name: str = None) -> Dict:
    """Get specific device profile or auto-detect."""
    detector = DeviceDetector()
    
    if profile_name:
        if profile_name not in detector.profiles:
            raise ValueError(f"Unknown profile: {profile_name}")
        return detector.profiles[profile_name]
    
    return detector.selected_profile or {}
