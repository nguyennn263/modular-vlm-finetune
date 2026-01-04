"""
Model Registry: Cho phép dễ dàng thay đổi và đăng ký models
"""
from typing import Dict, Type, Any, Optional
import torch.nn as nn


class ModelRegistry:
    """Registry pattern cho các model components"""
    _vision_encoders: Dict[str, Type[nn.Module]] = {}
    _projectors: Dict[str, Type[nn.Module]] = {}
    _llms: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_vision_encoder(cls, name: str):
        """Decorator để đăng ký vision encoder"""
        def decorator(model_cls: Type[nn.Module]):
            cls._vision_encoders[name] = model_cls
            return model_cls
        return decorator
    
    @classmethod
    def register_projector(cls, name: str):
        """Decorator để đăng ký projector"""
        def decorator(model_cls: Type[nn.Module]):
            cls._projectors[name] = model_cls
            return model_cls
        return decorator
    
    @classmethod
    def register_llm(cls, name: str, config: Dict[str, Any]):
        """Đăng ký LLM config"""
        cls._llms[name] = config
    
    @classmethod
    def get_vision_encoder(cls, name: str) -> Type[nn.Module]:
        if name not in cls._vision_encoders:
            raise ValueError(f"Vision encoder '{name}' not found. Available: {list(cls._vision_encoders.keys())}")
        return cls._vision_encoders[name]
    
    @classmethod
    def get_projector(cls, name: str) -> Type[nn.Module]:
        if name not in cls._projectors:
            raise ValueError(f"Projector '{name}' not found. Available: {list(cls._projectors.keys())}")
        return cls._projectors[name]
    
    @classmethod
    def get_llm_config(cls, name: str) -> Dict[str, Any]:
        if name not in cls._llms:
            raise ValueError(f"LLM '{name}' not found. Available: {list(cls._llms.keys())}")
        return cls._llms[name]
    
    @classmethod
    def list_vision_encoders(cls) -> list:
        return list(cls._vision_encoders.keys())
    
    @classmethod
    def list_projectors(cls) -> list:
        return list(cls._projectors.keys())
    
    @classmethod
    def list_llms(cls) -> list:
        return list(cls._llms.keys())


# Qwen2 family
ModelRegistry.register_llm("qwen2-0.5b", {
    "model_name": "Qwen/Qwen2-0.5B-Instruct",
    "hidden_size": 896,
    "max_length": 2048,
    "image_token_id": 151667,
})

ModelRegistry.register_llm("qwen2-1.5b", {
    "model_name": "Qwen/Qwen2-1.5B-Instruct",
    "hidden_size": 1536,
    "max_length": 2048,
    "image_token_id": 151667,
})

ModelRegistry.register_llm("qwen2-7b", {
    "model_name": "Qwen/Qwen2-7B-Instruct",
    "hidden_size": 3584,
    "max_length": 4096,
    "image_token_id": 151667,
})

# Vinallama / Vietnamese LLMs
ModelRegistry.register_llm("vinallama-2.7b", {
    "model_name": "vilm/vinallama-2.7b-chat",
    "hidden_size": 2560,
    "max_length": 2048,
    "image_token_id": 32000,
})

# Phi family (small & efficient)
ModelRegistry.register_llm("phi-2", {
    "model_name": "microsoft/phi-2",
    "hidden_size": 2560,
    "max_length": 2048,
    "image_token_id": 50296,
})


VISION_ENCODER_CONFIGS = {
    "internvit-300m": {
        "model_name": "OpenGVLab/InternViT-300M-448px",
        "hidden_size": 1024,
        "image_size": 448,
        "patch_size": 14,
    },
    "internvit-6b": {
        "model_name": "OpenGVLab/InternViT-6B-448px-V1-5",
        "hidden_size": 3200,
        "image_size": 448,
        "patch_size": 14,
    },
    "siglip-so400m": {
        "model_name": "google/siglip-so400m-patch14-384",
        "hidden_size": 1152,
        "image_size": 384,
        "patch_size": 14,
    },
    "clip-vit-l": {
        "model_name": "openai/clip-vit-large-patch14-336",
        "hidden_size": 1024,
        "image_size": 336,
        "patch_size": 14,
    },
}


def get_vision_config(name: str) -> Dict[str, Any]:
    """Get vision encoder config by name"""
    if name not in VISION_ENCODER_CONFIGS:
        raise ValueError(f"Vision encoder '{name}' not found. Available: {list(VISION_ENCODER_CONFIGS.keys())}")
    return VISION_ENCODER_CONFIGS[name]
