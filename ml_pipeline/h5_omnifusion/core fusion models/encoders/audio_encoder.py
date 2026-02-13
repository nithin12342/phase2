"""Audio encoder with Wav2Vec2 + eGeMAPS + Mamba."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np

from ..components.mamba_block import MambaEncoder


class AudioEncoder(nn.Module):
    """
    Audio encoder combining:
    - Wav2Vec2-Large: Deep acoustic representations
    - eGeMAPS: Expert prosodic features
    - Mamba: Efficient temporal modeling
    """
    
    def __init__(
        self,
        config,
        d_model: int = 384,
    ):
        super().__init__()
        
        self.config = config
        self.d_model = d_model
        self.use_pretrained = False  # Set to True if using raw audio
        
        self.wav2vec_dim = config.backbone_dim  # 1024
        self.egemaps_dim = config.egemaps_dim if config.use_egemaps else 0  # 88
        
        self.feature_dim = 768  # Pre-extracted SOTA features dimension
        
        self.input_proj = nn.Linear(self.feature_dim, d_model)
        
        self.use_egemaps = config.use_egemaps
        
        self.temporal_encoder = MambaEncoder(
            d_model=d_model,
            n_layers=config.n_mamba_layers,
            dropout=0.1,
        )
        
        self.summary_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        audio_features: torch.Tensor,
        audio_egemaps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            audio_features: Pre-extracted audio features (B, D) or (B, T, D)
            audio_egemaps: Optional eGeMAPS features (B, 88)
            
        Returns:
            Dict with 'sequence' (B, T, D) and 'summary' (B, D)
        """
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)
        
        batch_size, seq_len, feat_dim = audio_features.shape
        
        x = self.input_proj(audio_features)  # (B, T, d_model)
        
        x = self.temporal_encoder(x)  # (B, T, d_model)
        
        summary = x.mean(dim=1)  # (B, d_model)
        summary = self.summary_proj(summary)
        summary = self.norm(summary)
        
        return {
            'sequence': x,
            'summary': summary,
        }


class AudioEncoderRaw(nn.Module):
    """
    Audio encoder for raw waveform input using Wav2Vec2.
    Use this when processing raw audio rather than pre-extracted features.
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
            from transformers import Wav2Vec2Model
            self.wav2vec = Wav2Vec2Model.from_pretrained(
                config.backbone,
                output_hidden_states=True
            )
            
            if config.freeze_backbone:
                for param in self.wav2vec.parameters():
                    param.requires_grad = False
            
            self.extract_layer = config.extract_layer
            self.wav2vec_available = True
        except Exception as e:
            print(f"Warning: Could not load Wav2Vec2: {e}")
            self.wav2vec_available = False
            self.fallback_proj = nn.Linear(16000, d_model)
        
        self.wav2vec_dim = config.backbone_dim
        self.egemaps_dim = config.egemaps_dim if config.use_egemaps else 0
        total_dim = self.wav2vec_dim + self.egemaps_dim
        
        self.input_proj = nn.Linear(total_dim, d_model)
        
        self.temporal_encoder = MambaEncoder(
            d_model=d_model,
            n_layers=config.n_mamba_layers,
            dropout=0.1,
        )
        
        self.summary_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.use_egemaps = config.use_egemaps
    
    def forward(
        self, 
        audio_wav: torch.Tensor,
        audio_egemaps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with raw audio waveform.
        
        Args:
            audio_wav: Raw audio waveform (B, T_samples)
            audio_egemaps: Pre-computed eGeMAPS features (B, 88)
            
        Returns:
            Dict with 'sequence' (B, T, D) and 'summary' (B, D)
        """
        batch_size = audio_wav.shape[0]
        
        if self.wav2vec_available:
            with torch.no_grad() if self.config.freeze_backbone else torch.enable_grad():
                wav2vec_out = self.wav2vec(
                    audio_wav,
                    output_hidden_states=True,
                    return_dict=True
                )
                wav2vec_features = wav2vec_out.hidden_states[self.extract_layer]
        else:
            wav2vec_features = self.fallback_proj(audio_wav).unsqueeze(1)
        
        if self.use_egemaps and audio_egemaps is not None:
            egemaps_expanded = audio_egemaps.unsqueeze(1).expand(
                -1, wav2vec_features.shape[1], -1
            )
            combined = torch.cat([wav2vec_features, egemaps_expanded], dim=-1)
        else:
            combined = wav2vec_features
        
        x = self.input_proj(combined)
        
        x = self.temporal_encoder(x)
        
        summary = x.mean(dim=1)
        summary = self.summary_proj(summary)
        summary = self.norm(summary)
        
        return {
            'sequence': x,
            'summary': summary,
        }
