"""Face encoder with OpenFace AUs + LSTM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FaceEncoder(nn.Module):
    """
    Face encoder for Action Unit features:
    - 17 Action Units (intensity + presence)
    - Gaze features
    - Head pose
    - BiLSTM temporal modeling
    
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
        self.dino_dim = 384  # DINOv2-Small outputs 384-dim
        self.au_dim = config.au_dim  # 35 = 17*2 + gaze
        
        self.input_proj = nn.Linear(self.feature_dim, d_model)
        self.dino_proj = nn.Linear(self.dino_dim, d_model)
        self.au_proj = nn.Linear(self.au_dim, d_model)
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=config.n_lstm_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=0.1 if config.n_lstm_layers > 1 else 0,
        )
        
        self.summary_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        self.confidence_gate = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        face_features: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            face_features: Pre-extracted face features (B, D) or raw AUs (B, T, au_dim)
            confidence: Face detection confidence (B, T)
            
        Returns:
            Dict with 'sequence' and 'summary'
        """
        if face_features.dim() == 2:
            face_features = face_features.unsqueeze(1)
        
        B, T, feat_dim = face_features.shape
        
        if feat_dim == self.feature_dim:  # 768
            x = self.input_proj(face_features)
        elif feat_dim == self.dino_dim:  # 384
            x = self.dino_proj(face_features)
        else:
            x = self.au_proj(face_features)
        
        if confidence is not None:
            if confidence.dim() == 1:
                confidence = confidence.unsqueeze(1).expand(-1, T)
            conf_weight = self.confidence_gate(confidence.unsqueeze(-1))
            x = x * conf_weight
        
        x, (h_n, c_n) = self.lstm(x)  # (B, T, d_model)
        
        if self.config.bidirectional:
            summary = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            summary = h_n[-1]
        
        summary = self.summary_proj(summary)
        summary = self.norm(summary)
        
        return {
            'sequence': x,
            'summary': summary,
            'confidence_mean': confidence.mean() if confidence is not None else None,
        }
