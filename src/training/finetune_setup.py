"""
Clean fine-tuning setup for Vision-Language models.

Strategy:
- Freeze Vision Model completely
- Freeze Language Model completely  
- Train only the Bridge module
- Bridge converts Vision embeddings (4096 dims) → LLM embeddings (896 dims)
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

from src.modeling.bridge_modules import (
    LinearBridge,
    BetterMLP,
    MultiTokenMLP,
    AttentionBridge,
    MiniQFormer,
    QFormer
)


BRIDGE_TYPE = Literal['linear_bridge', 'better_mlp', 'multi_token', 'attention', 'mini_qformer', 'qformer']


class VisionLanguageBridge(nn.Module):
    """
    Unified wrapper for vision-language fine-tuning.
    
    Freezes both vision_model and language_model.
    Only trains the bridge module that converts:
    Vision embeddings (4096 dims) → LLM embeddings (896 dims)
    """
    
    def __init__(
        self,
        base_model,
        bridge_type: BRIDGE_TYPE,
        bridge_config: Optional[dict] = None
    ):
        super().__init__()
        
        self.bridge_type = bridge_type
        self.bridge_config = bridge_config or {}
        
        # Reference original models (will be frozen)
        self.vision_model = base_model.vision_model
        self.language_model = base_model.language_model
        
        # Create trainable bridge module
        self.bridge = self._create_bridge()
        
        # Freeze both models
        self._freeze_models()
    
    def _create_bridge(self) -> nn.Module:
        """Create bridge module based on type."""
        config = self.bridge_config
        
        # Vision encoder outputs 1024-dimensional embeddings (Vintern-1B)
        vision_dim = 1024
        # LLM expects 896-dimensional embeddings (Qwen2)
        hidden_dim = 896
        
        if self.bridge_type == 'linear_bridge':
            return LinearBridge(
                in_features=vision_dim,
                out_features=hidden_dim
            )
        
        elif self.bridge_type == 'better_mlp':
            return BetterMLP(
                in_features=vision_dim,
                out_features=hidden_dim
            )
        
        elif self.bridge_type == 'multi_token':
            return MultiTokenMLP(
                in_features=vision_dim,
                out_features=hidden_dim,
                num_tokens=config.get('num_tokens', 8)
            )
        
        elif self.bridge_type == 'attention':
            return AttentionBridge(
                vision_dim=vision_dim,
                hidden_dim=hidden_dim,
                num_tokens=config.get('num_tokens', 8),
                num_heads=config.get('num_heads', 8)
            )
        
        elif self.bridge_type == 'mini_qformer':
            return MiniQFormer(
                vision_dim=vision_dim,
                hidden_dim=hidden_dim,
                num_tokens=config.get('num_tokens', 8),
                num_heads=config.get('num_heads', 8)
            )
        
        elif self.bridge_type == 'qformer':
            return QFormer(
                vision_dim=vision_dim,
                hidden_dim=hidden_dim,
                num_queries=config.get('num_queries', 16),
                num_heads=config.get('num_heads', 8),
                num_layers=config.get('num_layers', 4)
            )
        
        else:
            raise ValueError(f"Unknown bridge_type: {self.bridge_type}")
    
    def _freeze_models(self):
        """Freeze vision and language models."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        for param in self.language_model.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pixel_values: Vision input (B, C, H, W) or (B, num_tiles, C, H, W)
            input_ids: Text tokens (B, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Logits or embeddings depending on use case
        """
        # Get vision embeddings (B, num_patches, 1024) [frozen]
        with torch.no_grad():
            vision_embeddings = self.vision_model(pixel_values)
        
        # Apply bridge (converts to LLM embedding space)
        if self.bridge_type in ['multi_token', 'attention', 'mini_qformer', 'qformer']:
            bridge_output = self.bridge(vision_embeddings)
        else:  # better_mlp, linear_bridge
            # Pool vision embeddings to single vector
            vision_pool = vision_embeddings.mean(dim=1)  # (B, 1024)
            bridge_output = self.bridge(vision_pool)  # (B, 896)
            bridge_output = bridge_output.unsqueeze(1)  # (B, 1, 896)
        
        # Get text embeddings (B, seq_len, 896) [frozen]
        with torch.no_grad():
            text_embeddings = self.language_model.model.embed_tokens(input_ids)
        
        # Combine vision and text embeddings
        combined_embeddings = torch.cat([bridge_output, text_embeddings], dim=1)
        
        # Create attention mask for combined embeddings
        if attention_mask is not None:
            vision_mask = torch.ones(
                combined_embeddings.shape[0],
                bridge_output.shape[1],
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        
        # Forward through LLM [frozen]
        with torch.no_grad():
            outputs = self.language_model(
                inputs_embeds=combined_embeddings,
                attention_mask=attention_mask,
                **kwargs
            )
        
        return outputs
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.bridge.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def create_finetune_model(
    base_model,
    bridge_type: BRIDGE_TYPE,
    bridge_config: Optional[dict] = None
) -> VisionLanguageBridge:
    """Factory function to create fine-tune model."""
    return VisionLanguageBridge(base_model, bridge_type, bridge_config)
