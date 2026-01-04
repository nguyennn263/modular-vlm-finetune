import torch
import torch.nn as nn
from typing import Optional

from .registry import ModelRegistry


@ModelRegistry.register_projector("mlp")
class MLPProjector(nn.Module):
    """
    Standard 2-layer MLP Projector
    Architecture: Linear -> GELU -> Linear
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim * 4
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


@ModelRegistry.register_projector("mlp_gelu")
class MLPGeLUProjector(nn.Module):
    """MLP with LayerNorm"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim * 4
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


@ModelRegistry.register_projector("identity")
class IdentityProjector(nn.Module):
    """Identity projector (no projection)"""
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        assert input_dim == output_dim, "Identity projector requires input_dim == output_dim"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@ModelRegistry.register_projector("linear")
class LinearProjector(nn.Module):
    """Simple linear projection"""
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@ModelRegistry.register_projector("downsample")
class DownsampleProjector(nn.Module):
    """Projector với pixel shuffle để giảm số tokens"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
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
        x: torch.Tensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        r = self.downsample_ratio
        
        if height is None or width is None:
            height = width = int(N ** 0.5)
        
        x = x.view(B, height, width, C)
        x = x.view(B, height // r, r, width // r, r, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, -1, C * r * r)
        
        return self.mlp(x)


def build_projector(
    projector_type: str,
    input_dim: int,
    output_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function để tạo projector
    Args:
        projector_type: "mlp", "mlp_gelu", "linear", "identity", "downsample"
        input_dim: Vision encoder hidden size
        output_dim: LLM hidden size
    """
    projector_cls = ModelRegistry.get_projector(projector_type)
    return projector_cls(input_dim=input_dim, output_dim=output_dim, **kwargs)
