"""MS² (Modality Shared-Specific) Decomposition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class MS2Decomposition(nn.Module):
    """
    Modality Shared-Specific (MS²) Decomposition
    
    Splits modality representations into:
    - Shared subspace: Cross-modal, depression-relevant features
    - Specific subspace: Modality-unique, potentially noisy features
    """
    
    def __init__(
        self,
        d_model: int = 384,
        shared_ratio: float = 0.5,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_shared = int(d_model * shared_ratio)
        self.d_specific = d_model - self.d_shared
        
        self.modality_names = ['audio', 'video', 'face', 'text', 'tabular']
        
        self.shared_projs = nn.ModuleDict({
            m: nn.Linear(d_model, self.d_shared)
            for m in self.modality_names
        })
        
        self.specific_projs = nn.ModuleDict({
            m: nn.Linear(d_model, self.d_specific)
            for m in self.modality_names
        })
        
        self.shared_norms = nn.ModuleDict({
            m: nn.LayerNorm(self.d_shared)
            for m in self.modality_names
        })
        
        self.specific_norms = nn.ModuleDict({
            m: nn.LayerNorm(self.d_specific)
            for m in self.modality_names
        })
    
    def forward(
        self,
        summaries: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        
        Args:
            summaries: Dict of modality summaries, each (B, D)
            
        Returns:
            Tuple of:
                - shared: Dict of (B, d_shared) tensors
                - specific: Dict of (B, d_specific) tensors
                - orth_loss: Orthogonality penalty
        """
        shared = {}
        specific = {}
        
        for m in self.modality_names:
            if m not in summaries:
                continue
            
            h = summaries[m]  # (B, D)
            
            s = self.shared_projs[m](h)
            q = self.specific_projs[m](h)
            
            s = self.shared_norms[m](s)
            q = self.specific_norms[m](q)
            
            shared[m] = s
            specific[m] = q
        
        orth_loss = self._compute_orthogonality_loss(shared, specific)
        
        return shared, specific, orth_loss
    
    def _compute_orthogonality_loss(
        self,
        shared: Dict[str, torch.Tensor],
        specific: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Penalize overlap between shared and specific subspaces."""
        loss = 0.0
        count = 0
        
        for m in self.modality_names:
            if m not in shared:
                continue
            
            s = shared[m]  # (B, d_shared)
            q = specific[m]  # (B, d_specific)
            
            min_dim = min(s.shape[1], q.shape[1])
            s_trunc = s[:, :min_dim]
            q_trunc = q[:, :min_dim]
            
            dot = torch.sum(s_trunc * q_trunc, dim=1)
            loss = loss + torch.mean(dot ** 2)
            count += 1
        
        if count > 0:
            return loss / count
        return torch.tensor(0.0, device=next(iter(shared.values())).device)
