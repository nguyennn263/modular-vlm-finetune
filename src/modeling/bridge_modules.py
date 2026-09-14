"""
Bridge modules for Vision-Language fine-tuning.

Philosophy: IMPROVE the baseline projection, don't replace it.
- Baseline: Linear projection (mimics Vintern's MLP1)
- Improvements: Add residuals, multi-token, attention, gating, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # Learnable tokens for query - use proper Xavier initialization
        self.queries = nn.Parameter(torch.empty(num_tokens, hidden_dim))
        # Xavier uniform initialization: std = sqrt(2 / (fan_in + fan_out))
        nn.init.xavier_uniform_(self.queries)
    
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
        
        # Self-attention: patches attend to each other. need_weights=False routes
        # nn.MultiheadAttention through the fused SDPA kernel, which does NOT
        # materialise the O(L^2) attention matrix -- required once L = n_tiles*P
        # grows (1536 patches at n_tiles=6 would otherwise need ~4.5 GiB).
        attn_out, _ = self.attention(
            baseline_patches, baseline_patches, baseline_patches, need_weights=False
        )

        # Residual + norm
        enhanced_patches = self.norm(baseline_patches + attn_out)  # (B, num_patches, 896)

        # Query with learnable tokens
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_tokens, 896)

        # Cross-attention: queries attend to enhanced patches
        output, _ = self.attention(queries, enhanced_patches, enhanced_patches, need_weights=False)
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
        
        # Gating - initialize with Xavier
        self.gate_fc = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.gate_fc.weight)
        nn.init.zeros_(self.gate_fc.bias)
    
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


class PatchPoolBridge(nn.Module):
    """
    Patch pooling bridge: mean- or max-pool the patch grid down to num_tokens.

    Philosophy: ISOLATE the pooling operator from everything else. Same first
    step as AttentionBridge (per-patch Linear(1024 -> 896)) so that the ONLY
    thing that differs between this and AttentionBridge is how patches -> tokens
    (fixed pooling operator vs. learned attention) -- not param count, not the
    per-patch projection. Deliberately has no other learnable weights: the point
    of this ablation is to isolate the pooling *operator*, not add capacity.

    Architecture:
    - Per-patch projection: Linear(1024 -> 896)
    - Pool num_patches -> num_tokens via F.adaptive_avg_pool1d / adaptive_max_pool1d
      over the sequence dim (robust to any patch/token count -- no manual
      reshape/grouping, avoids off-by-one edge cases when num_patches doesn't
      divide evenly by num_tokens)

    Note: F.adaptive_*_pool1d pools over the LAST dim, so the input must be
    transposed to (B, hidden_dim, num_patches) before pooling and back to
    (B, num_tokens, hidden_dim) after -- both transposes are load-bearing.
    """

    def __init__(self,
                 vision_dim: int = 1024,
                 hidden_dim: int = 896,
                 num_tokens: int = 8,
                 pool_type: str = "mean",
                 **kwargs):
        super().__init__()
        if pool_type not in ("mean", "max"):
            raise ValueError(f"pool_type must be 'mean' or 'max', got {pool_type!r}")
        self.pool_type = pool_type
        self.num_tokens = num_tokens
        self.proj = nn.Linear(vision_dim, hidden_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch_size, num_patches, vision_dim)

        Returns:
            (batch_size, num_tokens, hidden_dim)
        """
        x = self.proj(vision_features)  # (B, num_patches, hidden_dim)
        x = x.transpose(1, 2)  # (B, hidden_dim, num_patches) -- adaptive_*_pool1d pools the LAST dim
        pool_fn = F.adaptive_avg_pool1d if self.pool_type == "mean" else F.adaptive_max_pool1d
        x = pool_fn(x, self.num_tokens)  # (B, hidden_dim, num_tokens)
        return x.transpose(1, 2)  # (B, num_tokens, hidden_dim)


class _ConvResBlock(nn.Module):
    """3x3 conv residual block with GroupNorm (not BatchNorm -- batch_size=8
    makes BatchNorm running-stat estimates unstable/wrong at this scale)."""

    def __init__(self, dim: int, num_groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + residual)


class ConvAbstractorBridge(nn.Module):
    """
    Convolutional abstractor bridge, following HoneyBee's C-Abstractor
    (Cha et al., "Honeybee: Locality-enhanced Projector for Multimodal LLM",
    arXiv:2312.06742, CVPR 2024): "L ResNet blocks followed by adaptive average
    pooling and another L ResNet blocks."

    Philosophy: preserve LOCAL spatial context via convolution (unlike every
    other bridge in this file, which either discards spatial structure entirely
    -- MultiTokenMLP, sees a flat patch sequence with no 2D inductive bias --
    AttentionBridge/MiniQFormer/QFormer, or does content-agnostic pooling --
    PatchPoolBridge). Conv kernels only mix spatially-adjacent patches, so
    nearby image regions influence each other before/after the token count is
    reduced -- the "zoom in, compress, zoom out" design an advisor asked about.

    Architecture:
    - Reshape the flat patch sequence into a 2D (H, W) grid (requires a square
      patch count, or square+1 with a leading global token stripped -- true
      for InternViT/1-tile/336px: 24x24=576 patches + 1 leading token = 577)
    - 1x1 Conv2d projects vision_dim -> internal_dim
    - num_resblocks _ConvResBlock's (pre-pool)
    - AdaptiveAvgPool2d down to (grid, grid) where grid = sqrt(num_tokens)
      (num_tokens MUST be a perfect square -- e.g. 9 (3x3), 4 (2x2))
    - num_resblocks _ConvResBlock's (post-pool)
    - Flatten back to a token sequence + Linear(internal_dim -> hidden_dim)

    internal_dim defaults to 512 (not hidden_dim=896) to keep the param count
    on the same order as the other bridges (~20M at num_resblocks=2) rather
    than confounding "conv-based" with "biggest bridge in the study" (896-wide
    4x ResBlocks would be ~58M, next to Full-QFormer's 69M).
    """

    def __init__(self,
                 vision_dim: int = 1024,
                 hidden_dim: int = 896,
                 num_tokens: int = 9,
                 num_resblocks: int = 2,
                 internal_dim: int = 512,
                 num_groups: int = 32,
                 **kwargs):
        super().__init__()
        grid = int(round(num_tokens ** 0.5))
        if grid * grid != num_tokens:
            raise ValueError(f"ConvAbstractorBridge requires num_tokens to be a perfect "
                              f"square (e.g. 4, 9, 16), got {num_tokens}")
        self.grid = grid
        self.num_tokens = num_tokens

        self.proj_in = nn.Conv2d(vision_dim, internal_dim, kernel_size=1)
        self.pre = nn.ModuleList(_ConvResBlock(internal_dim, num_groups) for _ in range(num_resblocks))
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.post = nn.ModuleList(_ConvResBlock(internal_dim, num_groups) for _ in range(num_resblocks))
        self.proj_out = nn.Linear(internal_dim, hidden_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch_size, num_patches, vision_dim). num_patches
                must be a perfect square (a single tile's square patch grid),
                OR (num_patches - 1) a perfect square -- InternViT's raw
                `last_hidden_state` at 1-tile/336px is 577 = 24*24 + 1, i.e. a
                24x24 patch grid PLUS one leading global/CLS-like token
                (empirically confirmed: patch_size=14, image_size=336 for the
                standard training path, not the 448px figure used elsewhere in
                this file's docstrings for a *different* code path). The
                leading token is dropped before reshaping -- it has no 2D
                spatial position, so conv/pooling can't use it meaningfully;
                every other patch-based bridge in this file (AttentionBridge,
                MiniQFormer, QFormer) already implicitly includes it as if it
                were an ordinary patch when they pool/attend over the full
                sequence, so dropping it here is the ONE place in the codebase
                that treats it differently -- worth flagging in the eventual
                report, not silently glossed over.
                Multi-tile input (T*577, T>1) will fail loudly below unless T
                itself happens to make the (adjusted) count a perfect square --
                this bridge is single-tile-only by construction.

        Returns:
            (batch_size, num_tokens, hidden_dim)
        """
        B, P, D = vision_features.shape
        H = W = int(round(P ** 0.5))
        if H * W == P:
            x = vision_features
        elif int(round((P - 1) ** 0.5)) ** 2 == P - 1:
            H = W = int(round((P - 1) ** 0.5))
            x = vision_features[:, 1:, :]  # drop the leading global/CLS-like token
        else:
            raise ValueError(f"ConvAbstractorBridge requires a square patch grid (or square+1 "
                              f"with a leading global token), got num_patches={P} -- likely "
                              f"multi-tile input fed through a bridge that only supports a "
                              f"single tile's square grid")
        x = x.transpose(1, 2).reshape(B, D, H, W)  # (B, vision_dim, H, W), NCHW
        x = self.proj_in(x)
        for blk in self.pre:
            x = blk(x)
        x = self.pool(x)  # (B, internal_dim, grid, grid)
        for blk in self.post:
            x = blk(x)
        x = x.flatten(2).transpose(1, 2)  # (B, grid*grid, internal_dim)
        return self.proj_out(x)  # (B, num_tokens, hidden_dim)


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
        
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        ff_dim = hidden_dim * ff_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Self-attention on queries, then cross-attention to context."""
        # Self-attention
        attn_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + attn_out)
        
        # Cross-attention (queries to context)
        cross_attn_out, _ = self.cross_attn(queries, context, context)
        queries = self.norm2(queries + cross_attn_out)
        
        # FFN
        ffn_out = self.ffn(queries)
        queries = self.norm3(queries + ffn_out)
        
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
        # Initialize gate weights with Xavier initialization
        nn.init.xavier_uniform_(self.gate_fc.weight)
        nn.init.zeros_(self.gate_fc.bias)
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
        # Step 1: Vision cross-attention
        vision_attn, _ = self.vision_cross_attn(queries, vision_features, vision_features)
        queries_vision = self.vision_cross_norm(queries + vision_attn)
        
        # Step 2: Question cross-attention
        question_attn, _ = self.question_cross_attn(queries, question_embeddings, question_embeddings)
        queries_question = self.question_cross_norm(queries + question_attn)
        
        # Step 3: Gating - adaptive blend of vision-conditioned vs question-conditioned
        # Learn to balance: when to trust vision info vs semantic question info
        gate = torch.sigmoid(self.gate_fc(queries))  # (B, N, hidden_dim)
        # CORRECT: Blend two different representations using learned gate
        queries = gate * queries_vision + (1 - gate) * queries_question
        queries = self.gate_norm(queries)
        
        # Step 4: Self-attention - refine queries after gating
        self_attn, _ = self.self_attn(queries, queries, queries)
        queries = self.self_attn_norm(queries + self_attn)
        
        # Step 5: FFN
        ffn_out = self.ffn(queries)
        queries = self.ffn_norm(queries + ffn_out)
        
        return queries
