"""Stage 3: Latent Global Fusion with Perceiver architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from ..components.mamba_block import MambaBlock


class LatentFusionBlock(nn.Module):
    """Single block of latent fusion: Cross-attention + Mixing + FFN."""
    
    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        use_mamba: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.use_mamba = use_mamba
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)
        
        if use_mamba:
            self.latent_mix = MambaBlock(d_model, expand=1, dropout=dropout)
        else:
            self.latent_mix = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
        self.mix_norm = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            latents: (B, L, D)
            tokens: (B, T, D)
            
        Returns:
            Updated latents: (B, L, D)
        """
        attn_out, _ = self.cross_attn(latents, tokens, tokens)
        latents = self.cross_norm(latents + attn_out)
        
        if self.use_mamba:
            mix_out = self.latent_mix(latents)
            latents = mix_out  # Mamba includes residual
        else:
            mix_out, _ = self.latent_mix(latents, latents, latents)
            latents = self.mix_norm(latents + mix_out)
        
        ffn_out = self.ffn(latents)
        latents = self.ffn_norm(latents + ffn_out)
        
        return latents


class LatentGlobalFusion(nn.Module):
    """
    Latent Global Fusion (Stage 3)
    
    Uses Perceiver-style architecture with:
    - Learnable latent tokens
    - Cross-attention to all input tokens
    - Efficient O(L*T) complexity
    """
    
    def __init__(
        self,
        d_model: int = 384,
        n_latents: int = 32,
        n_blocks: int = 4,
        n_heads: int = 8,
        use_mamba: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.n_latents = n_latents
        self.d_model = d_model
        
        self.latent_init = nn.Parameter(
            torch.randn(n_latents, d_model) * 0.02
        )
        
        self.blocks = nn.ModuleList([
            LatentFusionBlock(
                d_model=d_model,
                n_heads=n_heads,
                use_mamba=use_mamba,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        ])
    
    def forward(
        self,
        fused_sequence: torch.Tensor,
        summaries: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            fused_sequence: (B, T, D) - output from Stage 1
            summaries: Dict of (B, D) modality summaries
            
        Returns:
            Tuple of:
                - z_CLS: (B, D) - global representation
                - latents: (B, L, D) - final latent states
        """
        B = fused_sequence.shape[0]
        
        latents = self.latent_init.unsqueeze(0).expand(B, -1, -1)
        
        modality_order = ['audio', 'video', 'face', 'text', 'tabular']
        summary_list = [summaries[m] for m in modality_order if m in summaries]
        
        if len(summary_list) > 0:
            summary_stack = torch.stack(summary_list, dim=1)  # (B, M, D)
            tokens = torch.cat([fused_sequence, summary_stack], dim=1)  # (B, T+M, D)
        else:
            tokens = fused_sequence
        
        for block in self.blocks:
            latents = block(latents, tokens)
        
        z_CLS = latents.mean(dim=1)  # (B, D)
        
        return z_CLS, latents
