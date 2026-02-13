"""Text encoder with MentalRoBERTa + Mamba/KAN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..components.mamba_block import MambaEncoder
from ..components.kan_layer import KAN


class TextEncoder(nn.Module):
    """
    Text encoder combining:
    - MentalRoBERTa: Mental health domain-adapted language model
    - Mamba/KAN: Efficient processing with interpretability
    
    Supports both pre-extracted features and raw text.
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
        
        self.temporal_encoder = MambaEncoder(
            d_model=d_model,
            n_layers=config.n_mamba_layers,
            dropout=0.1,
        )
        
        self.use_kan = config.use_kan
        if config.use_kan:
            self.kan = KAN(
                layer_dims=[d_model, d_model // 2, d_model],
                grid_size=5,
                spline_order=3,
            )
        
        self.summary_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            text_features: Pre-extracted text features (B, D) or (B, T, D)
            attention_mask: Optional mask (B, T)
            
        Returns:
            Dict with 'sequence' and 'summary'
        """
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        
        batch_size, seq_len, feat_dim = text_features.shape
        
        x = self.input_proj(text_features)  # (B, T, d_model)
        
        x = self.temporal_encoder(x)
        
        if self.use_kan:
            cls_token = x[:, 0, :]
            cls_enhanced = self.kan(cls_token)
            x = torch.cat([cls_enhanced.unsqueeze(1), x[:, 1:, :]], dim=1)
        
        summary = x[:, 0, :]
        summary = self.summary_proj(summary)
        summary = self.norm(summary)
        
        return {
            'sequence': x,
            'summary': summary,
        }


class TextEncoderRaw(nn.Module):
    """
    Text encoder for raw text input using MentalRoBERTa.
    Use this when processing raw text rather than pre-extracted features.
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
            from transformers import AutoModel, AutoTokenizer
            self.text_model = AutoModel.from_pretrained(config.backbone)
            self.tokenizer = AutoTokenizer.from_pretrained(config.backbone)
            
            if config.freeze_backbone:
                for param in self.text_model.parameters():
                    param.requires_grad = False
            
            self.model_available = True
        except Exception as e:
            print(f"Warning: Could not load {config.backbone}: {e}")
            self.model_available = False
            self.fallback_embed = nn.Embedding(30000, config.backbone_dim)
        
        self.input_proj = nn.Linear(config.backbone_dim, d_model)
        
        self.temporal_encoder = MambaEncoder(
            d_model=d_model,
            n_layers=config.n_mamba_layers,
            dropout=0.1,
        )
        
        self.use_kan = config.use_kan
        if config.use_kan:
            self.kan = KAN(
                layer_dims=[d_model, d_model // 2, d_model],
                grid_size=5,
                spline_order=3,
            )
        
        self.summary_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with tokenized text.
        
        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            
        Returns:
            Dict with 'sequence' and 'summary'
        """
        if self.model_available:
            with torch.no_grad() if self.config.freeze_backbone else torch.enable_grad():
                text_out = self.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                text_features = text_out.last_hidden_state
        else:
            text_features = self.fallback_embed(input_ids)
        
        x = self.input_proj(text_features)
        
        x = self.temporal_encoder(x)
        
        if self.use_kan:
            cls_token = x[:, 0, :]
            cls_enhanced = self.kan(cls_token)
            x = torch.cat([cls_enhanced.unsqueeze(1), x[:, 1:, :]], dim=1)
        
        summary = x[:, 0, :]
        summary = self.summary_proj(summary)
        summary = self.norm(summary)
        
        return {
            'sequence': x,
            'summary': summary,
        }
