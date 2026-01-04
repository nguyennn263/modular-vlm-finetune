"""
Vision Tower: InternViT wrapper cho VLM
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class VisionTower(nn.Module):
    """Vision Encoder sử dụng InternViT"""
    
    def __init__(
        self,
        model_name: str = "OpenGVLab/InternViT-300M-448px",
        freeze: bool = True,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
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
    
    def unfreeze(self):
        """Unfreeze model để full finetune"""
        for param in self.vision_model.parameters():
            param.requires_grad = True
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, num_patches, 3, H, W) hoặc (batch, 3, H, W)
        Returns:
            features: (batch, num_patches * seq_len, hidden_size) hoặc (batch, seq_len, hidden_size)
        """
        # Handle both 4D và 5D input
        if pixel_values.dim() == 4:
            # Single image per batch: (B, 3, H, W)
            is_batched = False
            B, C, H, W = pixel_values.shape
            N = 1
        elif pixel_values.dim() == 5:
            # Multiple patches per sample: (B, N, 3, H, W)
            is_batched = True
            B, N, C, H, W = pixel_values.shape
            # Flatten batch và patches
            pixel_values = pixel_values.view(B * N, C, H, W)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {pixel_values.dim()}D")
        
        with torch.set_grad_enabled(self.training):
            outputs = self.vision_model(pixel_values, output_hidden_states=True)
        
        features = outputs.last_hidden_state  # (B*N, seq_len, hidden) or (B, seq_len, hidden)
        
        # Reshape về batch
        if is_batched:
            seq_len = features.shape[1]
            features = features.view(B, N * seq_len, -1)
        
        return features
    
    @property
    def dtype(self):
        return next(self.vision_model.parameters()).dtype
    
    @property
    def device(self):
        return next(self.vision_model.parameters()).device
