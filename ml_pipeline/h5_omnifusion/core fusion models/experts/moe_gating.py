"""Quality-aware Mixture-of-Experts gating."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .modality_experts import ModalityExpert, FusionExpert


class QualityFeatureExtractor(nn.Module):
    """Extract quality features for gating."""
    
    def __init__(self, n_features: int = 5):
        super().__init__()
        self.n_features = n_features
    
    def forward(
        self,
        audio_quality: torch.Tensor,
        face_confidence: torch.Tensor,
        text_length: torch.Tensor,
        video_motion: torch.Tensor,
        tabular_completeness: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute quality features.
        
        Returns:
            quality: (B, n_features)
        """
        audio_q = audio_quality.float()
        if audio_q.dim() > 1:
            audio_q = audio_q.mean(dim=-1)
        
        face_q = face_confidence.float()
        if face_q.dim() > 1:
            face_q = face_q.mean(dim=-1)
        
        text_q = text_length.float() / 500.0  # Normalize
        if text_q.dim() > 1:
            text_q = text_q.mean(dim=-1)
        
        video_q = video_motion.float()
        if video_q.dim() > 1:
            video_q = video_q.mean(dim=-1)
        
        tabular_q = tabular_completeness.float()
        if tabular_q.dim() > 1:
            tabular_q = tabular_q.mean(dim=-1)
        
        all_quality = [
            audio_q,
            face_q,
            text_q,
            video_q,
            tabular_q,
        ]
        
        quality = torch.stack(all_quality[:self.n_features], dim=-1)
        
        return quality


class QualityAwareGate(nn.Module):
    """Quality-aware gating network for MoE."""
    
    def __init__(
        self,
        d_model: int = 384,
        n_experts: int = 6,
        n_quality_features: int = 5,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        gate_input_dim = 5 * d_model + d_model + n_quality_features
        
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_experts),
        )
        
        self.d_model = d_model
    
    def forward(
        self,
        summaries: Dict[str, torch.Tensor],
        z_CLS: torch.Tensor,
        quality: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gate weights.
        
        Returns:
            Tuple of:
                - weights: (B, n_experts) normalized weights
                - logits: (B, n_experts) raw logits
        """
        B = z_CLS.shape[0]
        device = z_CLS.device
        
        zero = torch.zeros(B, self.d_model, device=device)
        
        gate_input = torch.cat([
            summaries.get('audio', zero),
            summaries.get('video', zero),
            summaries.get('face', zero),
            summaries.get('text', zero),
            summaries.get('tabular', zero),
            z_CLS,
            quality,
        ], dim=-1)
        
        logits = self.gate_network(gate_input)
        weights = F.softmax(logits, dim=-1)
        
        return weights, logits


class MixtureOfExperts(nn.Module):
    """
    Complete Mixture of Experts module.
    
    Combines modality-specific experts with quality-aware gating.
    """
    
    def __init__(
        self,
        d_model: int = 384,
        d_shared: int = 192,
        d_specific: int = 192,
        expert_hidden: int = 128,
        n_quality_features: int = 5,
    ):
        super().__init__()
        
        self.d_model = d_model
        expert_input_dim = d_shared + d_specific
        
        self.audio_expert = ModalityExpert(expert_input_dim, expert_hidden)
        self.video_expert = ModalityExpert(expert_input_dim, expert_hidden)
        self.face_expert = ModalityExpert(expert_input_dim, expert_hidden)
        self.text_expert = ModalityExpert(expert_input_dim, expert_hidden)
        self.tabular_expert = ModalityExpert(expert_input_dim, expert_hidden)
        
        self.fusion_expert = FusionExpert(d_model, 256)
        
        self.gate = QualityAwareGate(
            d_model=d_model,
            n_experts=6,
            n_quality_features=n_quality_features,
        )
        
        self.quality_extractor = QualityFeatureExtractor(n_quality_features)
        
        self.fallback_dim = expert_input_dim
    
    def forward(
        self,
        shared: Dict[str, torch.Tensor],
        specific: Dict[str, torch.Tensor],
        z_CLS: torch.Tensor,
        summaries: Dict[str, torch.Tensor],
        quality_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of:
                - final_logit: (B, 1) weighted prediction
                - gate_weights: (B, 6) expert weights
                - expert_logits: (B, 6) individual predictions
        """
        B = z_CLS.shape[0]
        device = z_CLS.device
        
        expert_inputs = {}
        for m in ['audio', 'video', 'face', 'text', 'tabular']:
            if m in shared and m in specific:
                expert_inputs[m] = torch.cat([shared[m], specific[m]], dim=-1)
        
        zero_input = torch.zeros(B, self.fallback_dim, device=device)
        
        expert_outputs = [
            self.audio_expert(expert_inputs.get('audio', zero_input)),
            self.video_expert(expert_inputs.get('video', zero_input)),
            self.face_expert(expert_inputs.get('face', zero_input)),
            self.text_expert(expert_inputs.get('text', zero_input)),
            self.tabular_expert(expert_inputs.get('tabular', zero_input)),
            self.fusion_expert(z_CLS),
        ]
        expert_logits = torch.cat(expert_outputs, dim=-1)  # (B, 6)
        
        quality = self.quality_extractor(
            quality_inputs.get('audio_quality', torch.ones(B, device=device)),
            quality_inputs.get('face_confidence', torch.ones(B, device=device)),
            quality_inputs.get('text_length', torch.ones(B, device=device) * 100),
            quality_inputs.get('video_motion', torch.ones(B, device=device) * 0.5),
            quality_inputs.get('tabular_completeness', torch.ones(B, device=device)),
        )
        
        gate_weights, gate_logits = self.gate(summaries, z_CLS, quality)
        
        final_logit = torch.sum(gate_weights * expert_logits, dim=1, keepdim=True)
        
        return final_logit, gate_weights, expert_logits
