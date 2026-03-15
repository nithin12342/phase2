"""Stage 2: Modality-level hypergraph attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ModalityLevelHypergraph(nn.Module):
    """
    Modality-Level Hypergraph (Stage 2a)
    
    Allows modality summaries to exchange information
    through transformer-style attention.
    """
    
    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        
        self.modality_names = ['audio', 'video', 'face', 'text', 'tabular']
    
    def forward(
        self,
        summaries: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            summaries: Dict of modality summaries, each (B, D)
            
        Returns:
            Updated summaries
        """
        available = [m for m in self.modality_names if m in summaries]
        
        if len(available) == 0:
            return summaries
        
        summary_list = [summaries[m] for m in available]
        summary_stack = torch.stack(summary_list, dim=1)
        
        updated = self.transformer(summary_stack)  # (B, M, D)
        
        updated_summaries = {}
        for i, m in enumerate(available):
            updated_summaries[m] = updated[:, i, :]
        
        return updated_summaries
