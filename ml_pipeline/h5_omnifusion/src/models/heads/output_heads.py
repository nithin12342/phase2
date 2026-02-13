"""Multi-task output heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiTaskOutputHead(nn.Module):
    """
    Multi-task output heads for:
    - Binary depression classification
    - PHQ-8 score regression
    - Optional VAD prediction
    """
    
    def __init__(
        self,
        d_model: int = 384,
        n_classes: int = 2,
        phq_range: tuple = (0, 24),
        predict_vad: bool = False,
    ):
        super().__init__()
        
        self.phq_range = phq_range
        self.predict_vad = predict_vad
        
        self.phq_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        
        if predict_vad:
            self.vad_head = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 3),  # Valence, Arousal, Dominance
            )
    
    def forward(
        self,
        final_logit: torch.Tensor,
        z_CLS: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            final_logit: (B, 1) from MoE
            z_CLS: (B, D) global representation
            
        Returns:
            Dict with predictions
        """
        outputs = {}
        
        outputs['binary_logit'] = final_logit
        outputs['binary_prob'] = torch.sigmoid(final_logit)
        
        phq_score = self.phq_head(z_CLS)
        phq_score = torch.clamp(phq_score, self.phq_range[0], self.phq_range[1])
        outputs['phq_score'] = phq_score
        
        if self.predict_vad:
            vad = self.vad_head(z_CLS)
            outputs['valence'] = torch.tanh(vad[:, 0:1])
            outputs['arousal'] = torch.tanh(vad[:, 1:2])
            outputs['dominance'] = torch.tanh(vad[:, 2:3])
        
        return outputs
