"""
Vision Encoders với Registry support
Dễ dàng thêm/thay đổi vision encoder
"""
import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel, AutoConfig, CLIPVisionModel, SiglipVisionModel

from .registry import ModelRegistry


class BaseVisionEncoder(nn.Module):
    """Base class cho tất cả vision encoders"""
    
    def __init__(self, hidden_size: int, image_size: int = 448):
        super().__init__()
        self.hidden_size = hidden_size
        self.image_size = image_size
        self._frozen = False
    
    def freeze(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
        self._frozen = True
    
    def unfreeze(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        self._frozen = False
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        return next(self.parameters()).device


@ModelRegistry.register_vision_encoder("internvit")
class InternViTEncoder(BaseVisionEncoder):
    """InternViT Vision Encoder"""
    
    def __init__(
        self,
        model_name: str = "OpenGVLab/InternViT-300M-448px",
        hidden_size: int = 1024,
        image_size: int = 448,
        use_flash_attention: bool = False,
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__(hidden_size, image_size)
        
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Flash attention nếu được hỗ trợ
        if use_flash_attention:
            config._attn_implementation = "flash_attention_2"
        
        self.model = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self.hidden_size = config.hidden_size
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W) hoặc (B, N, C, H, W)
        Returns:
            features: (B, seq_len, hidden_size)
        """
        if pixel_values.dim() == 5:
            B, N, C, H, W = pixel_values.shape
            pixel_values = pixel_values.view(B * N, C, H, W)
            outputs = self.model(pixel_values, output_hidden_states=True)
            features = outputs.last_hidden_state
            seq_len = features.shape[1]
            features = features.view(B, N * seq_len, -1)
        else:
            outputs = self.model(pixel_values, output_hidden_states=True)
            features = outputs.last_hidden_state
        
        return features


@ModelRegistry.register_vision_encoder("siglip")
class SiglipEncoder(BaseVisionEncoder):
    """SigLIP Vision Encoder - efficient và hiệu quả"""
    
    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        hidden_size: int = 1152,
        image_size: int = 384,
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__(hidden_size, image_size)
        
        self.model = SiglipVisionModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        self.hidden_size = self.model.config.hidden_size
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() == 5:
            B, N, C, H, W = pixel_values.shape
            pixel_values = pixel_values.view(B * N, C, H, W)
            outputs = self.model(pixel_values)
            features = outputs.last_hidden_state
            seq_len = features.shape[1]
            features = features.view(B, N * seq_len, -1)
        else:
            outputs = self.model(pixel_values)
            features = outputs.last_hidden_state
        
        return features


@ModelRegistry.register_vision_encoder("clip")
class CLIPEncoder(BaseVisionEncoder):
    """CLIP Vision Encoder"""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14-336",
        hidden_size: int = 1024,
        image_size: int = 336,
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__(hidden_size, image_size)
        
        self.model = CLIPVisionModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        self.hidden_size = self.model.config.hidden_size
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() == 5:
            B, N, C, H, W = pixel_values.shape
            pixel_values = pixel_values.view(B * N, C, H, W)
            outputs = self.model(pixel_values)
            features = outputs.last_hidden_state
            seq_len = features.shape[1]
            features = features.view(B, N * seq_len, -1)
        else:
            outputs = self.model(pixel_values)
            features = outputs.last_hidden_state
        
        return features


def build_vision_encoder(
    encoder_type: str,
    model_name: Optional[str] = None,
    **kwargs
) -> BaseVisionEncoder:
    """
    Factory function để tạo vision encoder
    
    Args:
        encoder_type: "internvit", "siglip", "clip"
        model_name: Override model name
        **kwargs: Additional arguments
    Returns:
        Vision encoder instance
    """
    encoder_cls = ModelRegistry.get_vision_encoder(encoder_type)
    
    if model_name:
        kwargs["model_name"] = model_name
    
    return encoder_cls(**kwargs)
