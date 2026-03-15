"""Tabular encoder with KAN/TabTransformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..components.kan_layer import KAN


class TabularEncoder(nn.Module):
    """
    Tabular encoder for clinical metadata:
    - Demographics
    - PHQ item scores
    - Interview statistics
    - Behavioral features
    """
    
    def __init__(
        self,
        config,
        d_model: int = 384,
    ):
        super().__init__()
        
        self.config = config
        self.d_model = d_model
        self.n_features = config.n_features
        
        self.feature_dim = 768  # If using pre-extracted
        
        self.feature_proj = nn.Linear(self.feature_dim, d_model)
        
        embed_dim = max(d_model // config.n_features, 16)
        self.feature_embeds = nn.ModuleList([
            nn.Linear(1, embed_dim)
            for _ in range(config.n_features)
        ])
        self.raw_proj = nn.Linear(embed_dim * config.n_features, d_model)
        
        self.use_kan = config.use_kan
        if config.use_kan:
            self.processor = KAN(
                layer_dims=[d_model, d_model // 2, d_model],
                grid_size=config.kan_grid_size,
                spline_order=config.kan_spline_order,
            )
        else:
            self.processor = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, d_model),
            )
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        tabular_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            tabular_features: Tabular data (B, n_features) or (B, D) pre-extracted
            
        Returns:
            Dict with 'summary' (no sequence for tabular)
        """
        if tabular_features.dim() == 3:
            tabular_features = tabular_features.squeeze(1)
            
        B = tabular_features.shape[0]
        feat_dim = tabular_features.shape[-1]
        
        if feat_dim == self.feature_dim:
            x = self.feature_proj(tabular_features)
        elif feat_dim == self.n_features:
            embeds = []
            for i, embed_layer in enumerate(self.feature_embeds):
                feat = tabular_features[:, i:i+1]
                embed = embed_layer(feat)
                embeds.append(embed)
            x = torch.cat(embeds, dim=-1)
            x = self.raw_proj(x)
        else:
            fallback_proj = nn.Linear(feat_dim, self.d_model).to(tabular_features.device)
            x = fallback_proj(tabular_features)
        
        x = self.processor(x)
        
        summary = self.output_proj(x)
        summary = self.norm(summary)
        
        return {
            'summary': summary,
        }
