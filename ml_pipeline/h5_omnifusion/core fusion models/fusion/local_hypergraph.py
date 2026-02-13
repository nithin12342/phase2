"""Stage 1: Local Hypergraph Fusion for time-aligned cross-modal attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple


class LocalHypergraphFusion(nn.Module):
    """
    Local Hypergraph Fusion (Stage 1)
    
    At each time step, performs multi-head attention over all modalities
    to capture simultaneous cross-modal patterns.
    """
    
    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        modality_sequences: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            modality_sequences: Dict with aligned sequences
                Each tensor: (B, T, D)
                
        Returns:
            Tuple of:
                - fused_sequence: (B, T, D)
                - attention_weights: (B, T, n_heads, M, M)
        """
        modality_list = ['audio', 'video', 'face', 'text', 'tabular']
        sequences = []
        for m in modality_list:
            if m in modality_sequences:
                sequences.append(modality_sequences[m])
        
        if len(sequences) == 0:
            raise ValueError("No modality sequences provided")
        
        stacked = torch.stack(sequences, dim=2)
        B, T, M, D = stacked.shape
        
        stacked_flat = stacked.view(B * T, M, D)
        
        Q = self.W_q(stacked_flat).view(B*T, M, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(stacked_flat).view(B*T, M, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(stacked_flat).view(B*T, M, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B*T, M, D)
        attn_output = self.W_o(attn_output)
        
        attn_output = self.norm(stacked_flat + attn_output)
        
        fused = attn_output.mean(dim=1)
        
        fused_sequence = fused.view(B, T, D)
        
        attn_weights = attn_weights.view(B, T, self.n_heads, M, M)
        
        return fused_sequence, attn_weights
