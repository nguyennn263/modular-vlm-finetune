"""
Bridge modules for Vision-Language fine-tuning.
These modules bridge vision embeddings to language model input embeddings.
"""

import torch
import torch.nn as nn
import math


class LinearBridge(nn.Module):
    """
    Simplest bridge: single linear projection.
    
    Architecture:
    - Linear(4096 → 896)
    
    Rationale:
    - Baseline for ablation study
    - Shows importance of bridge complexity
    - Minimal parameters for comparison
    - Fastest inference and training
    - No hidden layers, no activation functions
    """
    
    def __init__(self, in_features: int = 1024, out_features: int = 896, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, in_features) or (batch_size, in_features)
        
        Returns:
            (batch_size, seq_len, out_features) or (batch_size, out_features)
        """
        return self.fc(x)


class BetterMLP(nn.Module):
    """
    Improved MLP with residual connection for vision-to-language bridging.
    
    Architecture:
    - LayerNorm(4096)
    - Linear(4096 → 2048)
    - GELU
    - Linear(2048 → 896)
    - Skip connection: Linear(4096 → 896)
    - Output = main + skip
    
    Rationale:
    - LayerNorm stabilizes training with large vision features (4096)
    - Bottleneck (4096→2048) reduces computational cost
    - Skip connection improves gradient flow and training stability
    - Residual allows learning incremental changes vs direct projection
    """
    
    def __init__(self, in_features: int = 4096, out_features: int = 896, **kwargs):
        super().__init__()
        hidden_dim = 2048
        
        # Main path
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_features)
        
        # Skip connection with projection
        self.skip = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, in_features) or (batch_size, in_features)
        
        Returns:
            (batch_size, seq_len, out_features) or (batch_size, out_features)
        """
        # Main path
        main = self.norm(x)
        main = self.fc1(main)
        main = self.act(main)
        main = self.fc2(main)
        
        # Skip path
        skip = self.skip(x)
        
        # Combine
        return main + skip


class MultiTokenMLP(nn.Module):
    """
    Generate multiple tokens to represent vision features.
    
    Input: (B, 4096) from pooled vision features
    Output: (B, N, 896) where N is number of tokens
    
    Architecture:
    - Linear(4096 → 896*N)
    - Reshape to (B, N, 896)
    
    Rationale:
    - Single token creates bottleneck (1 vector for all visual info)
    - Multiple tokens increase representation capacity
    - Each token can specialize in different visual aspects
    - Allows better fusion with text tokens
    """
    
    def __init__(self, in_features: int = 4096, out_features: int = 896, num_tokens: int = 8, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_tokens = num_tokens
        
        self.fc = nn.Linear(in_features, out_features * num_tokens)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_features)
        
        Returns:
            (batch_size, num_tokens, out_features)
        """
        B = x.shape[0]
        x = self.fc(x)  # (B, out_features*num_tokens)
        x = x.reshape(B, self.num_tokens, self.out_features)  # (B, N, 896)
        return x


class AttentionBridge(nn.Module):
    """
    Use multi-head attention to compress vision features to language model input.
    
    Architecture:
    - Project vision embeddings: Linear(1024 → 896)
    - Learnable queries: nn.Parameter (N, 896)
    - Multi-head attention: queries attend to vision features
    - Output: (B, N, 896)
    
    Rationale:
    - Attention allows dynamic selection of relevant visual features
    - Multiple heads capture different visual aspects independently
    - Learnable queries serve as "information slots" for vision data
    - Soft selection is differentiable and learns with task
    """
    
    def __init__(self, 
                 vision_dim: int = 1024, 
                 hidden_dim: int = 896, 
                 num_tokens: int = 8,
                 num_heads: int = 8,
                 **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        
        # Project vision features to hidden_dim
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_tokens, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch_size, num_patches, vision_dim)
        
        Returns:
            (batch_size, num_tokens, hidden_dim)
        """
        B = vision_features.shape[0]
        
        # Project vision features
        vision_proj = self.vision_proj(vision_features)  # (B, num_patches, 896)
        
        # Expand learnable queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, N, 896)
        
        # Multi-head attention: queries attend to vision features
        # query=queries, key=value=vision_proj
        attn_out, _ = self.attention(queries, vision_proj, vision_proj)  # (B, N, 896)
        
        # Layer norm
        output = self.norm(attn_out + queries)
        
        return output


class MiniQFormer(nn.Module):
    """
    Mini Q-Former with 2 transformer layers for lightweight vision-language bridging.
    Based on BLIP-2 Q-Former but much smaller.
    
    Architecture:
    - Learnable queries: (N, 896), N=8
    - 2 Transformer layers, each with:
      - Self-attention on queries
      - Cross-attention (queries ↔ vision)
      - Feed-forward network
      - Residual connections, layer normalization
    
    Output: (B, N, 896)
    
    Rationale:
    - Self-attention allows tokens to communicate and refine representations
    - Cross-attention enables vision-to-query information flow
    - 2 layers balance expressiveness vs computational cost
    - Skip connections improve optimization
    """
    
    def __init__(self,
                 vision_dim: int = 1024,
                 hidden_dim: int = 896,
                 num_tokens: int = 8,
                 num_heads: int = 8,
                 ff_multiplier: int = 4,
                 **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        
        # Project vision features
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_tokens, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)
        
        # 2 Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, ff_multiplier)
            for _ in range(2)
        ])
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch_size, num_patches, vision_dim)
        
        Returns:
            (batch_size, num_tokens, hidden_dim)
        """
        B = vision_features.shape[0]
        
        # Project vision features
        vision_proj = self.vision_proj(vision_features)  # (B, num_patches, 896)
        
        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, N, 896)
        
        # Pass through transformer layers
        for layer in self.layers:
            queries = layer(queries, vision_proj)
        
        return queries


class QFormer(nn.Module):
    """
    Full Q-Former for advanced vision-language bridging.
    Conditions vision features on question embeddings for semantic filtering.
    
    Architecture:
    - Project vision: Linear(1024 → 896)
    - Learnable queries: (N, 896), N=16
    - 4 QFormer layers, each with:
      - Cross-attention: queries ↔ vision (extract visual info)
      - Cross-attention: queries ↔ question embeddings (condition)
      - Gating: fuse vision and question attention outputs adaptively
      - Self-attention: refine query representations
      - Feed-forward
      - Full residual + layer norm
    
    Output: (B, 16, 896)
    
    Rationale:
    - Vision cross-attn extracts relevant visual features
    - Question cross-attn provides semantic context/filtering
    - Gating allows adaptive weighting of vision vs question signals:
      - High gate = trust vision features
      - Low gate = trust question guidance
    - Self-attention enables inter-token communication
    - 4 layers allow progressive refinement
    - Compared to BLIP-2: fewer layers, no pretraining, but same concepts
    """
    
    def __init__(self,
                 vision_dim: int = 1024,
                 hidden_dim: int = 896,
                 num_queries: int = 16,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 ff_multiplier: int = 4,
                 **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Project vision features
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)
        
        # Stack of QFormer layers
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, num_heads, ff_multiplier)
            for _ in range(num_layers)
        ])
    
    def forward(self, 
                vision_features: torch.Tensor,
                question_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch_size, num_patches, vision_dim)
            question_embeddings: (batch_size, question_len, hidden_dim)
        
        Returns:
            (batch_size, num_queries, hidden_dim)
        """
        B = vision_features.shape[0]
        
        # Project vision features
        vision_proj = self.vision_proj(vision_features)  # (B, num_patches, 896)
        
        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, N, 896)
        
        # Pass through QFormer layers
        for layer in self.layers:
            queries = layer(queries, vision_proj, question_embeddings)
        
        return queries


# ============= Helper Layers =============

class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and FFN."""
    
    def __init__(self, hidden_dim: int, num_heads: int, ff_multiplier: int):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        ff_dim = hidden_dim * ff_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Self-attention on queries, then cross-attention to context."""
        # Self-attention
        attn_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + attn_out)
        
        # Cross-attention (queries to context)
        cross_attn_out, _ = self.self_attn(queries, context, context)
        queries = self.norm1(queries + cross_attn_out)
        
        # FFN
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)
        
        return queries


class QFormerLayer(nn.Module):
    """Q-Former layer with vision+question conditioning and gating."""
    
    def __init__(self, hidden_dim: int, num_heads: int, ff_multiplier: int):
        super().__init__()
        
        # Cross-attention: queries ↔ vision
        self.vision_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.vision_cross_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention: queries ↔ question
        self.question_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.question_cross_norm = nn.LayerNorm(hidden_dim)
        
        # Gating mechanism for adaptive fusion
        self.gate_fc = nn.Linear(hidden_dim, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        
        # Self-attention: refine queries
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward
        ff_dim = hidden_dim * ff_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self,
                queries: torch.Tensor,
                vision_features: torch.Tensor,
                question_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Step 1: Cross-attention with vision
        Step 2: Cross-attention with question
        Step 3: Gating - adaptive fusion of vision vs question
        Step 4: Self-attention - refine queries
        Step 5: FFN
        """
        residual = queries
        
        # Step 1: Vision cross-attention
        vision_attn, _ = self.vision_cross_attn(queries, vision_features, vision_features)
        vision_attn = self.vision_cross_norm(residual + vision_attn)
        
        # Step 2: Question cross-attention
        question_attn, _ = self.question_cross_attn(queries, question_embeddings, question_embeddings)
        question_attn = self.question_cross_norm(residual + question_attn)
        
        # Step 3: Gating - combine vision and question info
        gate = torch.sigmoid(self.gate_fc(queries))  # (B, N, hidden_dim)
        fused = gate * vision_attn + (1 - gate) * question_attn  # Adaptive blending
        fused = self.gate_norm(fused)
        
        # Step 4: Self-attention
        self_attn, _ = self.self_attn(fused, fused, fused)
        fused = self.self_attn_norm(fused + self_attn)
        
        # Step 5: FFN
        ffn_out = self.ffn(fused)
        output = self.ffn_norm(fused + ffn_out)
        
        return output
