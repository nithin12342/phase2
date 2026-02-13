"""Video encoder with ViT + TimeSformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from einops import rearrange

from ..components.mamba_block import MambaEncoder


class TimeSformerBlock(nn.Module):
    """
    TimeSformer block with divided space-time attention.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(d_model)
        
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
        x: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Forward pass with divided space-time attention.
        
        Args:
            x: Input (B, T*P, D) where T=frames, P=patches
            num_frames: Number of temporal frames
            
        Returns:
            Output (B, T*P, D)
        """
        B, N, D = x.shape
        T = num_frames
        P = N // T if T > 0 else N
        
        if T > 1 and P > 1:
            x_t = rearrange(x, 'b (t p) d -> (b p) t d', t=T)
            t_out, _ = self.temporal_attn(x_t, x_t, x_t)
            t_out = rearrange(t_out, '(b p) t d -> b (t p) d', b=B)
            x = self.temporal_norm(x + t_out)
            
            x_s = rearrange(x, 'b (t p) d -> (b t) p d', t=T)
            s_out, _ = self.spatial_attn(x_s, x_s, x_s)
            s_out = rearrange(s_out, '(b t) p d -> b (t p) d', b=B)
            x = self.spatial_norm(x + s_out)
        else:
            attn_out, _ = self.temporal_attn(x, x, x)
            x = self.temporal_norm(x + attn_out)
        
        x = self.ffn_norm(x + self.ffn(x))
        
        return x


class VideoEncoder(nn.Module):
    """
    Video encoder combining:
    - ViT: Frame-level visual features
    - TimeSformer: Temporal dynamics
    - Mamba: Efficient sequence modeling
    
    Supports pre-extracted features.
    """
    
    def __init__(
        self,
        config,
        d_model: int = 384,
    ):
        super().__init__()
        
        self.config = config
        self.d_model = d_model
        
        self.feature_dim = 768
        
        self.input_proj = nn.Linear(self.feature_dim, d_model)
        
        self.use_timesformer = config.use_timesformer
        if config.use_timesformer:
            self.timesformer = TimeSformerBlock(
                d_model=d_model,
                num_heads=8,
                dropout=0.1,
            )
        
        self.temporal_encoder = MambaEncoder(
            d_model=d_model,
            n_layers=config.n_mamba_layers,
            dropout=0.1,
        )
        
        self.summary_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        video_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            video_features: Pre-extracted video features (B, D) or (B, T, D)
            
        Returns:
            Dict with 'sequence' and 'summary'
        """
        if video_features.dim() == 2:
            video_features = video_features.unsqueeze(1)
        
        B, T, feat_dim = video_features.shape
        
        x = self.input_proj(video_features)  # (B, T, d_model)
        
        if self.use_timesformer and T > 1:
            x = self.timesformer(x, num_frames=T)
        
        x = self.temporal_encoder(x)
        
        summary = x.mean(dim=1)
        summary = self.summary_proj(summary)
        summary = self.norm(summary)
        
        return {
            'sequence': x,
            'summary': summary,
        }


class VideoEncoderRaw(nn.Module):
    """
    Video encoder for raw video frames using ViT.
    Use this when processing raw frames rather than pre-extracted features.
    """
    
    def __init__(
        self,
        config,
        d_model: int = 384,
    ):
        super().__init__()
        
        self.config = config
        self.d_model = d_model
        
        try:
            import timm
            self.vit = timm.create_model(
                config.backbone.split('/')[-1],
                pretrained=True,
                num_classes=0,
            )
            
            if config.freeze_backbone:
                for param in self.vit.parameters():
                    param.requires_grad = False
            
            self.vit_available = True
        except Exception as e:
            print(f"Warning: Could not load ViT: {e}")
            self.vit_available = False
            self.fallback_conv = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, config.backbone_dim)
            )
        
        self.input_proj = nn.Linear(config.backbone_dim, d_model)
        
        self.use_timesformer = config.use_timesformer
        if config.use_timesformer:
            self.timesformer = TimeSformerBlock(
                d_model=d_model,
                num_heads=8,
                dropout=0.1,
            )
        
        self.temporal_encoder = MambaEncoder(
            d_model=d_model,
            n_layers=config.n_mamba_layers,
            dropout=0.1,
        )
        
        self.summary_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        video_frames: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with raw video frames.
        
        Args:
            video_frames: Video frames (B, T, C, H, W)
            
        Returns:
            Dict with 'sequence' and 'summary'
        """
        B, T, C, H, W = video_frames.shape
        
        frames_flat = video_frames.view(B * T, C, H, W)
        
        if self.vit_available:
            with torch.no_grad() if self.config.freeze_backbone else torch.enable_grad():
                frame_features = self.vit(frames_flat)
        else:
            frame_features = self.fallback_conv(frames_flat)
        
        frame_features = frame_features.view(B, T, -1)
        
        x = self.input_proj(frame_features)
        
        if self.use_timesformer:
            x = self.timesformer(x, num_frames=T)
        
        x = self.temporal_encoder(x)
        
        summary = x.mean(dim=1)
        summary = self.summary_proj(summary)
        summary = self.norm(summary)
        
        return {
            'sequence': x,
            'summary': summary,
        }
