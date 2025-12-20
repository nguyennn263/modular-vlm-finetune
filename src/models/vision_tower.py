"""
Vision Tower: InternViT wrapper cho VLM
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class VisionTower(nn.Module):
    """Vision Encoder sử dụng InternViT hoặc tương tự"""
    
    def __init__(
        self,
        model_name: str = "OpenGVLab/InternViT-300M-448px",
        freeze: bool = True,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
        # Load vision model
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if use_flash_attention:
            config._attn_implementation = "flash_attention_2"
        
        self.vision_model = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        self.hidden_size = config.hidden_size
        
        if freeze:
            self._freeze_model()
    
    def _freeze_model(self):
        """Freeze tất cả parameters"""
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, num_patches, 3, H, W)
        Returns:
            features: (batch, total_tokens, hidden_size)
        """
        B, N, C, H, W = pixel_values.shape
        
        # Flatten batch và patches
        pixel_values = pixel_values.view(B * N, C, H, W)
        
        # Forward qua vision model
        outputs = self.vision_model(pixel_values)
        features = outputs.last_hidden_state  # (B*N, seq_len, hidden)
        
        # Reshape về batch
        seq_len = features.shape[1]
        features = features.view(B, N * seq_len, -1)
        
        return features
    
    @property
    def dtype(self):
        return next(self.vision_model.parameters()).dtype
    
    @property
    def device(self):
        return next(self.vision_model.parameters()).device
