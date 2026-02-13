"""Complete H⁵-OmniFusion model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .encoders.audio_encoder import AudioEncoder
from .encoders.text_encoder import TextEncoder
from .encoders.video_encoder import VideoEncoder
from .encoders.face_encoder import FaceEncoder
from .encoders.tabular_encoder import TabularEncoder
from .fusion.local_hypergraph import LocalHypergraphFusion
from .fusion.modality_hypergraph import ModalityLevelHypergraph
from .fusion.ms2_decomposition import MS2Decomposition
from .fusion.latent_perceiver import LatentGlobalFusion
from .experts.moe_gating import MixtureOfExperts
from .heads.output_heads import MultiTaskOutputHead


class H5OmniFusion(nn.Module):
    """
    H⁵-OmniFusion: Hypergraph-Hybrid-Hierarchical-High-Order OmniFusion
    
    Complete multimodal architecture for depression detection combining:
    - Stage 0: Unimodal temporal encoders (Mamba/KAN)
    - Stage 1: Local hypergraph fusion (time-aligned)
    - Stage 2: Modality-level hypergraph + MS² decomposition
    - Stage 3: Latent global fusion (Perceiver)
    - Stage 4: Quality-aware MoE gating
    - Stage 5: Multi-task output heads
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        d_model = config.d_model
        
        self.audio_encoder = AudioEncoder(config.audio, d_model)
        self.text_encoder = TextEncoder(config.text, d_model)
        self.video_encoder = VideoEncoder(config.video, d_model)
        self.face_encoder = FaceEncoder(config.face, d_model)
        self.tabular_encoder = TabularEncoder(config.tabular, d_model)
        
        self.local_hypergraph = LocalHypergraphFusion(
            d_model=d_model,
            n_heads=config.fusion.local_n_heads,
            dropout=config.fusion.local_dropout,
        )
        
        self.modality_hypergraph = ModalityLevelHypergraph(
            d_model=d_model,
            n_heads=config.fusion.modality_n_heads,
            n_layers=config.fusion.modality_n_layers,
            dropout=config.dropout,
        )
        
        self.ms2 = MS2Decomposition(
            d_model=d_model,
            shared_ratio=config.fusion.shared_ratio,
        ) if config.fusion.use_ms2 else None
        
        self.latent_fusion = LatentGlobalFusion(
            d_model=d_model,
            n_latents=config.fusion.n_latents,
            n_blocks=config.fusion.n_perceiver_blocks,
            n_heads=config.fusion.perceiver_n_heads,
            use_mamba=config.fusion.use_mamba_in_perceiver,
            dropout=config.dropout,
        )
        
        d_shared = int(d_model * config.fusion.shared_ratio) if config.fusion.use_ms2 else d_model // 2
        d_specific = d_model - d_shared if config.fusion.use_ms2 else d_model // 2
        
        self.moe = MixtureOfExperts(
            d_model=d_model,
            d_shared=d_shared,
            d_specific=d_specific,
            expert_hidden=config.moe.expert_hidden_dim,
            n_quality_features=config.moe.n_quality_features,
        )
        
        self.output_head = MultiTaskOutputHead(
            d_model=d_model,
            n_classes=config.n_classes,
            predict_vad=config.predict_vad,
        )
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Complete forward pass.
        
        Args:
            batch: Dictionary containing:
                - audio_features: Pre-extracted audio (B, D) or (B, T, D)
                - text_features: Pre-extracted text (B, D) or (B, T, D)
                - video_features: Pre-extracted video (B, D) or (B, T, D)
                - face_features: Pre-extracted face/image (B, D) or (B, T, D)
                - tabular_features: Tabular data (B, D)
                - quality_inputs: Optional dict for MoE gating
                - targets: Dict with 'binary', 'phq_score' (for training)
                
        Returns:
            Tuple of:
                - outputs: Dict with predictions
                - orth_loss: Orthogonality penalty
        """
        device = next(self.parameters()).device
        
        audio_out = self.audio_encoder(
            batch.get('audio_features', torch.zeros(1, 768, device=device)),
        )
        
        text_out = self.text_encoder(
            batch.get('text_features', torch.zeros(1, 768, device=device)),
        )
        
        video_out = self.video_encoder(
            batch.get('video_features', torch.zeros(1, 768, device=device)),
        )
        
        face_out = self.face_encoder(
            batch.get('face_features', torch.zeros(1, 768, device=device)),
        )
        
        tabular_out = self.tabular_encoder(
            batch.get('tabular_features', torch.zeros(1, 768, device=device)),
        )
        
        sequences = {
            'audio': audio_out['sequence'],
            'text': text_out['sequence'],
            'video': video_out['sequence'],
            'face': face_out['sequence'],
        }
        
        summaries = {
            'audio': audio_out['summary'],
            'text': text_out['summary'],
            'video': video_out['summary'],
            'face': face_out['summary'],
            'tabular': tabular_out['summary'],
        }
        
        aligned_sequences = self._align_sequences(sequences, summaries)
        
        fused_sequence, local_attn = self.local_hypergraph(aligned_sequences)
        
        updated_summaries = self.modality_hypergraph(summaries)
        
        if self.ms2 is not None:
            shared, specific, orth_loss = self.ms2(updated_summaries)
        else:
            shared = {m: s[:, :s.shape[1]//2] for m, s in updated_summaries.items()}
            specific = {m: s[:, s.shape[1]//2:] for m, s in updated_summaries.items()}
            orth_loss = torch.tensor(0.0, device=fused_sequence.device)
        
        z_CLS, latents = self.latent_fusion(fused_sequence, updated_summaries)
        
        quality_inputs = batch.get('quality_inputs', {})
        final_logit, gate_weights, expert_logits = self.moe(
            shared, specific, z_CLS, updated_summaries, quality_inputs
        )
        
        outputs = self.output_head(final_logit, z_CLS)
        
        outputs['gate_weights'] = gate_weights
        outputs['expert_logits'] = expert_logits
        outputs['local_attention'] = local_attn
        outputs['latents'] = latents
        
        return outputs, orth_loss
    
    def _align_sequences(
        self,
        sequences: Dict[str, torch.Tensor],
        summaries: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Align all sequences to common temporal dimension."""
        ref_len = 1
        for seq in sequences.values():
            if seq.shape[1] > ref_len:
                ref_len = seq.shape[1]
                break
        
        aligned = {}
        for m, seq in sequences.items():
            if seq.shape[1] != ref_len:
                seq = seq.transpose(1, 2)  # (B, D, T)
                seq = F.interpolate(seq, size=ref_len, mode='linear', align_corners=False)
                seq = seq.transpose(1, 2)  # (B, T, D)
            aligned[m] = seq
        
        if 'tabular' in summaries:
            aligned['tabular'] = summaries['tabular'].unsqueeze(1).expand(-1, ref_len, -1)
        
        return aligned
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
