"""
Bridge modules for Vision-Language fine-tuning.

Philosophy: IMPROVE the baseline projection, don't replace it.
- Baseline: Linear projection (mimics Vintern's MLP1)
- Improvements: Add residuals, multi-token, attention, gating, etc.
"""

import torch
import torch.nn as nn
import math


class LinearBridgeBaseline(nn.Module):
    """
    Baseline: single linear projection (mimics Vintern's MLP1).
    
    Architecture:
    - Linear(1024 → 896)
    
    Purpose:
    - Baseline projection from vision to LLM space
    - Used as foundation for all improvements
    - Minimal parameters, fast inference
    """
    
    def __init__(self, in_features: int = 1024, out_features: int = 896, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input to output space."""
        return self.fc(x)


# ============= IMPROVEMENT-BASED BRIDGES =============
# All use residual: output = baseline(x) + improvement(x)


class ResidualBridge(nn.Module):
    """
    Residual improvement over baseline linear projection.
    
    Architecture:
    - Baseline: Linear(1024 → 896)
    - Improvement: LayerNorm → Linear(1024 → 2048) → GELU → Linear(2048 → 896)
    - Output: baseline(x) + improvement(x)
    
    Benefits:
    - Keeps baseline alignment intact
    - Learns "adjustment" instead of replacement
    - Stable training with residual connections
    """
    
    def __init__(self, in_features: int = 1024, out_features: int = 896, **kwargs):
        super().__init__()
        hidden_dim = 2048
        
        # Baseline (frozen would be better, but we'll train it)
        self.baseline = nn.Linear(in_features, out_features)
        
        # Improvement path
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Baseline
        baseline_out = self.baseline(x)
        
        # Improvement
        improvement = self.norm(x)
        improvement = self.fc1(improvement)
        improvement = self.act(improvement)
        improvement = self.fc2(improvement)
        
        # Residual
        return baseline_out + improvement


class LinearBridge(nn.Module):
    """Legacy alias for ResidualBridge (maintains compatibility)."""
    
    def __init__(self, in_features: int = 1024, out_features: int = 896, **kwargs):
        super().__init__()
        self.bridge = ResidualBridge(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bridge(x)


class BetterMLP(nn.Module):
    """Legacy alias for ResidualBridge (maintains compatibility)."""
    
    def __init__(self, in_features: int = 4096, out_features: int = 896, **kwargs):
        super().__init__()
        # BetterMLP is for pooled vision features (4096)
        # Map down to 1024 first (simulating vision_dim)
        self.vision_proj = nn.Linear(in_features, 1024)
        self.bridge = ResidualBridge(1024, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vision_proj(x)
        return self.bridge(x)


class MultiTokenMLP(nn.Module):
    """
    Multi-token improvement over baseline projection.
    
    Philosophy: IMPROVE baseline by adding multiple query tokens
    - Baseline: single token Linear(1024 → 896)
    - Improvement: generate k additional tokens via Linear(1024 → 896*k)
    - Output: baseline_token + improvement_tokens (stacked)
    
    Architecture:
    - Baseline: Linear(1024 → 896) outputs shape (B, 896)
    - Improvement: Linear(1024 → 896*(k-1)) outputs shape (B, 896*(k-1))
    - Combined: (B, k, 896)
    
    Benefits:
    - Keeps baseline alignment as anchor token
    - Additional tokens learn complementary aspects
    - Gradual capacity increase (k=2 is minimal, k=8 is richer)
    """
    
    def __init__(self, in_features: int = 1024, out_features: int = 896, num_tokens: int = 8, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_tokens = num_tokens
        
        # Baseline token
        self.baseline = nn.Linear(in_features, out_features)
        
        # Improvement tokens (num_tokens - 1)
        num_improvement_tokens = max(num_tokens - 1, 1)
        self.improvement = nn.Linear(in_features, out_features * num_improvement_tokens)
        
        self.num_improvement_tokens = num_improvement_tokens
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_features)
        
        Returns:
            (batch_size, num_tokens, out_features)
        """
        B = x.shape[0]
        
        # Baseline token
        baseline_token = self.baseline(x)  # (B, 896)
        baseline_token = baseline_token.unsqueeze(1)  # (B, 1, 896)
        
        # Improvement tokens
        improvement_tokens = self.improvement(x)  # (B, 896 * (num_tokens-1))
        improvement_tokens = improvement_tokens.reshape(B, self.num_improvement_tokens, self.out_features)  # (B, num_tokens-1, 896)
        
        # Combine: baseline as anchor + improvements
        output = torch.cat([baseline_token, improvement_tokens], dim=1)  # (B, num_tokens, 896)
        
        return output


class AttentionBridge(nn.Module):
    """
    Tile Attention: Use self-attention to model relationships between vision patches.
    
    Philosophy: IMPROVE baseline with spatial awareness
    - Compute baseline projection for each patch
    - Apply self-attention to understand patch interactions
    - Aggregate with attention-weighted pool
    
    Architecture:
    - Baseline: Linear(1024 → 896) applied to each patch
    - Self-attention: patches attend to each other
    - Weighted aggregation: combine patches using attention weights
    - Output: (B, 896) single token with spatial awareness
    
    Benefits:
    - Baseline applies individually to each patch (no interaction)
    - Self-attention learns which patches matter
    - Differentiable sorting of visual importance
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
        
        # Baseline projection (applied to each patch)
        self.baseline = nn.Linear(vision_dim, hidden_dim)
        
        # Self-attention to model patch relationships
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Optional: learnable tokens for query
        self.queries = nn.Parameter(torch.randn(num_tokens, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch_size, num_patches, vision_dim)
        
        Returns:
            (batch_size, num_tokens, hidden_dim)
        """
        B = vision_features.shape[0]
        
        # Baseline: project each patch
        baseline_patches = self.baseline(vision_features)  # (B, num_patches, 896)
        
        # Self-attention: patches attend to each other
        attn_out, _ = self.attention(
            baseline_patches,  # query
            baseline_patches,  # key
            baseline_patches   # value
        )
        
        # Residual + norm
        enhanced_patches = self.norm(baseline_patches + attn_out)  # (B, num_patches, 896)
        
        # Query with learnable tokens
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_tokens, 896)
        
        # Cross-attention: queries attend to enhanced patches
        output, _ = self.attention(queries, enhanced_patches, enhanced_patches)
        output = self.norm(queries + output)
        
        return output


class MiniQFormer(nn.Module):
    """
    Lightweight Q-Former: Improved baseline with 2-layer transformer.
    
    Philosophy: IMPROVE baseline with minimal layers
    - Baseline: single linear projection  
    - Improvement: learnable queries + 2 transformer layers
    - Output: concatenate baseline + improvement tokens
    
    Architecture:
    - Baseline: Linear(1024 → 896)
    - Learnable queries: (4, 896) - fewer queries, minimal complexity
    - 2 Transformer layers for spatial reasoning
    
    Benefits:
    - Keeps baseline as anchor
    - Lightweight (only 2 layers vs 4 in full QFormer)
    - Good balance: expressiveness vs efficiency
    - 4 queries enough for most vision tasks
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
        self.num_tokens = max(num_tokens - 1, 1)  # Reserve 1 for baseline
        
        # Baseline projection
        self.baseline = nn.Linear(vision_dim, hidden_dim)
        
        # Improvement: project vision and learnable queries
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        # Learnable queries for improvement
        self.improvement_queries = nn.Parameter(torch.randn(self.num_tokens, hidden_dim))
        nn.init.normal_(self.improvement_queries, std=0.02)
        
        # 2 Transformer layers for refinement
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, ff_multiplier)
            for _ in range(2)
        ])
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch_size, num_patches, vision_dim)
        
        Returns:
            (batch_size, num_tokens+1, hidden_dim)
        """
        B = vision_features.shape[0]
        
        # Baseline token
        baseline_token = self.baseline(vision_features.mean(dim=1))  # (B, 896)
        baseline_token = baseline_token.unsqueeze(1)  # (B, 1, 896)
        
        # Improvement: project vision patches
        vision_proj = self.vision_proj(vision_features)  # (B, num_patches, 896)
        
        # Improvement queries
        queries = self.improvement_queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_tokens, 896)
        
        # Pass through transformer layers
        for layer in self.layers:
            queries = layer(queries, vision_proj)
        
        # Combine: baseline + improvement tokens
        output = torch.cat([baseline_token, queries], dim=1)  # (B, 1+num_tokens, 896)
        
        return output


class GatedFusionBridge(nn.Module):
    """
    Gated residual improvement for stable enhancement.
    
    Philosophy: IMPROVE baseline with learnable gating
    - Baseline: Linear(1024 → 896)
    - Improvement: deeper net
    - Output: baseline + gate * improvement (adaptive blending)
    
    Architecture:
    - Baseline: Linear(1024 → 896)
    - Improvement path: LayerNorm → 2 layers → gating sigmoid
    - Gate: learned per-element: when to use baseline vs improvement
    
    Benefits:
    - Prevents saturation: gate learns optimal blend
    - High gate = trust improvement, Low gate = keep baseline
    - More stable than simple residual (prevents divergence)
    """
    
    def __init__(self, in_features: int = 1024, out_features: int = 896, **kwargs):
        super().__init__()
        hidden_dim = 2048
        
        # Baseline
        self.baseline = nn.Linear(in_features, out_features)
        
        # Improvement path
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_features)
        
        # Gating
        self.gate_fc = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adaptive blend: output = baseline + gate * improvement
        """
        # Baseline
        baseline = self.baseline(x)
        
        # Improvement
        improvement = self.norm(x)
        improvement = self.fc1(improvement)
        improvement = self.act(improvement)
        improvement = self.fc2(improvement)
        
        # Gating: learn when to apply improvement
        gate = torch.sigmoid(self.gate_fc(x))  # (B, 896)
        
        # Adaptive blend
        output = baseline + gate * improvement
        
        return output


class TileAttentionBridge(nn.Module):
    """
    Tile Attention with spatial awareness (alias for AttentionBridge)."""
    
    def __init__(self, 
                 vision_dim: int = 1024, 
                 hidden_dim: int = 896,
                 num_tokens: int = 8,
                 num_heads: int = 8,
                 **kwargs):
        super().__init__()
        self.bridge = AttentionBridge(vision_dim, hidden_dim, num_tokens, num_heads)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.bridge(vision_features)


class QFormer(nn.Module):
    """
    Full Q-Former with 4 layers for advanced vision-language bridging.
    Improved: Uses residual pattern with baseline + improvement.
    
    Philosophy: IMPROVE baseline with semantic filtering
    - Baseline: single linear projection of pooled vision
    - Improvement: queries + 4-layer transformer with vision+text fusion
    - Output: concatenate baseline + queries
    
    Architecture:
    - Baseline: Linear(1024 → 896) from pooled vision
    - Improvement:
      * Learnable queries: (8, 896)
      * 4 QFormer layers with:
        - Cross-attention: queries ↔ vision (extract visual info)
        - Cross-attention: queries ↔ text (semantic filtering)
        - Gating: adaptive fusion of vision vs text
        - Self-attention: refine queries
        - FFN
    - Output: (B, 1+8, 896) = baseline + improvement queries
    
    Benefits:
    - Baseline ensures alignment with original projection
    - Improvement learns semantic context from text
    - Gating prevents over-reliance on text or vision
    - 4 layers for progressive refinement
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
        self.num_queries = max(num_queries - 1, 1)  # Reserve 1 for baseline
        
        # Baseline projection (single token)
        self.baseline = nn.Linear(vision_dim, hidden_dim)
        
        # Improvement: project vision + text
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        # Learnable queries for improvement
        self.improvement_queries = nn.Parameter(torch.randn(self.num_queries, hidden_dim))
        nn.init.normal_(self.improvement_queries, std=0.02)
        
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
            (batch_size, 1+num_queries, hidden_dim)
        """
        B = vision_features.shape[0]
        
        # Baseline token
        baseline_token = self.baseline(vision_features.mean(dim=1))  # (B, 896)
        baseline_token = baseline_token.unsqueeze(1)  # (B, 1, 896)
        
        # Improvement: project vision
        vision_proj = self.vision_proj(vision_features)  # (B, num_patches, 896)
        
        # Improvement queries
        queries = self.improvement_queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, 896)
        
        # Pass through QFormer layers
        for layer in self.layers:
            queries = layer(queries, vision_proj, question_embeddings)
        
        # Combine baseline + improvement queries
        output = torch.cat([baseline_token, queries], dim=1)  # (B, 1+num_queries, 896)
        
        return output


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
