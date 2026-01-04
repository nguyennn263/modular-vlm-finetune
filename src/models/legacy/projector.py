"""
MLP Projector: Kết nối Vision Tower với LLM
2-layer MLP với GELU activation
"""
import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """
    Architecture: Linear -> GELU -> Linear
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 2048,
        hidden_dim: int = None,
    ):
        super().__init__()
        
        # Hidden dim mặc định = 4 * input_dim (theo paper)
        if hidden_dim is None:
            hidden_dim = input_dim * 4
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights với small std"""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch, num_patches, vision_dim)
        Returns:
            projected: (batch, num_patches, llm_dim)
        """
        return self.mlp(vision_features)


class DownsampleProjector(nn.Module):
    """
    Projector với pixel shuffle để giảm số tokens
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 2048,
        downsample_ratio: int = 2,
    ):
        super().__init__()
        
        self.downsample_ratio = downsample_ratio
        merged_dim = input_dim * (downsample_ratio ** 2)
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(merged_dim),
            nn.Linear(merged_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(
        self, 
        vision_features: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Args:
            vision_features: (batch, num_patches, vision_dim)
            height, width: spatial dimensions
        Returns:
            projected: (batch, reduced_patches, llm_dim)
        """
        B, N, C = vision_features.shape
        r = self.downsample_ratio
        
        # Reshape để merge spatial tokens
        x = vision_features.view(B, height, width, C)
        x = x.view(B, height // r, r, width // r, r, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, -1, C * r * r)
        
        return self.mlp(x)
