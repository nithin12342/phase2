import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class ModalityExpert(nn.Module):
    """Simple MLP expert for a single modality."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FusionExpert(nn.Module):
    """Expert that looks at the fused CLS token."""
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QualityFeatureExtractor(nn.Module):
    """Projects raw quality scores into a learned embedding space."""
    def __init__(self, n_features: int = 5):
        super().__init__()
        self.proj = nn.Linear(n_features, n_features)
    def forward(self, audio_q, face_c, text_l, video_m, tab_c):
        q = torch.stack([audio_q, face_c, text_l, video_m, tab_c], dim=-1)
        if q.dim() == 1: q = q.unsqueeze(0)
        return self.proj(q)

class QualityAwareGate(nn.Module):
    """Gating network that considers modality summaries and quality scores."""
    def __init__(self, d_model: int, n_experts: int, n_quality_features: int):
        super().__init__()
        self.d_model = d_model
        input_dim = (d_model * 6) + n_quality_features
        self.gate_network = nn.Linear(input_dim, n_experts)
        
    def forward(self, summaries: Dict[str, torch.Tensor], z_CLS: torch.Tensor, quality: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        gate_input = torch.clamp(gate_input, min=-100.0, max=100.0)
        logits = self.gate_network(gate_input)
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        weights = F.softmax(logits, dim=-1)
        return weights, logits

class MixtureOfExperts(nn.Module):
    def __init__(self, d_model: int = 384, d_shared: int = 192, d_specific: int = 192, expert_hidden: int = 128, n_quality_features: int = 5):
        super().__init__()
        self.d_model = d_model
        expert_input_dim = d_shared + d_specific
        self.audio_expert = ModalityExpert(expert_input_dim, expert_hidden)
        self.video_expert = ModalityExpert(expert_input_dim, expert_hidden)
        self.face_expert = ModalityExpert(expert_input_dim, expert_hidden)
        self.text_expert = ModalityExpert(expert_input_dim, expert_hidden)
        self.tabular_expert = ModalityExpert(expert_input_dim, expert_hidden)
        self.fusion_expert = FusionExpert(d_model, 256)
        self.gate = QualityAwareGate(d_model, 6, n_quality_features)
        self.quality_extractor = QualityFeatureExtractor(n_quality_features)
        self.fallback_dim = expert_input_dim

    def forward(self, shared, specific, z_CLS, summaries, quality_inputs):
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
        expert_logits = torch.cat(expert_outputs, dim=-1)
        
        quality = self.quality_extractor(
            quality_inputs.get('audio_quality', torch.ones(B, device=device)),
            quality_inputs.get('face_confidence', torch.ones(B, device=device)),
            quality_inputs.get('text_length', torch.ones(B, device=device) * 100),
            quality_inputs.get('video_motion', torch.ones(B, device=device) * 0.5),
            quality_inputs.get('tabular_completeness', torch.ones(B, device=device)),
        )
        gate_weights, _ = self.gate(summaries, z_CLS, quality)
        expert_logits = torch.clamp(expert_logits, min=-15.0, max=15.0)
        final_logit = torch.sum(gate_weights * expert_logits, dim=1, keepdim=True)
        return final_logit, gate_weights, expert_logits
