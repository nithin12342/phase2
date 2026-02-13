"""Model configuration for H⁵-OmniFusion."""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ComputeTier(Enum):
    """Compute tier for different resource constraints."""
    NANO = "nano"       # ~34K params, for tiny datasets (200 samples)
    MICRO = "micro"     # ~0.65M params, for medium datasets (400-800 samples)
    MEDIUM = "medium"   # ~4.5M params, for large datasets (1000+ samples)


@dataclass
class AudioEncoderConfig:
    """Audio encoder configuration."""
    backbone: str = "facebook/wav2vec2-large-xlsr-53"
    use_egemaps: bool = True
    egemaps_dim: int = 88
    backbone_dim: int = 1024
    extract_layer: int = 8  # Intermediate layer for prosody
    freeze_backbone: bool = True
    n_mamba_layers: int = 2
    

@dataclass
class TextEncoderConfig:
    """Text encoder configuration."""
    backbone: str = "mental/mental-roberta-base"
    backbone_dim: int = 768
    max_length: int = 512
    freeze_backbone: bool = False  # Fine-tune for depression
    n_mamba_layers: int = 1
    use_kan: bool = True


@dataclass
class VideoEncoderConfig:
    """Video encoder configuration."""
    backbone: str = "MCG-NJU/videomae-base"  # VideoMAE for temporal understanding
    backbone_dim: int = 768
    frame_rate: int = 1  # Frames per second
    freeze_backbone: bool = True
    use_timesformer: bool = True
    n_mamba_layers: int = 2


@dataclass
class FaceEncoderConfig:
    """Face encoder configuration."""
    use_openface: bool = True
    au_dim: int = 35  # 17 AUs × 2 + gaze features
    emotion_dim: int = 256  # Emotion model output
    n_lstm_layers: int = 2
    bidirectional: bool = True


@dataclass
class TabularEncoderConfig:
    """Tabular encoder configuration."""
    n_features: int = 20
    use_kan: bool = True
    kan_grid_size: int = 5
    kan_spline_order: int = 3


@dataclass
class FusionConfig:
    """Fusion module configuration."""
    local_n_heads: int = 2
    local_dropout: float = 0.5
    
    modality_n_heads: int = 2
    modality_n_layers: int = 1
    
    use_ms2: bool = False
    shared_ratio: float = 0.5  # Fraction for shared subspace
    
    n_latents: int = 4
    n_perceiver_blocks: int = 1
    perceiver_n_heads: int = 2
    use_mamba_in_perceiver: bool = False


@dataclass
class MoEConfig:
    """Mixture of Experts configuration."""
    n_experts: int = 6  # 5 modality + 1 fusion
    expert_hidden_dim: int = 32
    gate_hidden_dim: int = 64
    n_quality_features: int = 5
    use_quality_gating: bool = True


@dataclass
class LossConfig:
    """Loss function configuration."""
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    lambda_cls: float = 1.0
    lambda_phq: float = 0.5
    lambda_orth: float = 0.1
    decision_threshold: float = 0.35


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    warmup_ratio: float = 0.1


@dataclass
class H5Config:
    """Complete H⁵-OmniFusion configuration."""
    d_model: int = 16
    dropout: float = 0.5
    
    audio: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    video: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)
    face: FaceEncoderConfig = field(default_factory=FaceEncoderConfig)
    tabular: TabularEncoderConfig = field(default_factory=TabularEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    n_epochs: int = 20
    patience: int = 5
    
    n_classes: int = 2  # Binary depression
    phq_range: tuple = (0, 24)
    predict_vad: bool = False
    
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    max_grad_norm: float = 1.0
    max_seq_len: int = 256
    
    @classmethod
    def from_tier(cls, tier: ComputeTier) -> "H5Config":
        """Create configuration from compute tier."""
        configs = {
            ComputeTier.NANO: {
                "d_model": 16,
                "dropout": 0.4,
                "fusion": FusionConfig(
                    local_n_heads=2,
                    local_dropout=0.4,
                    modality_n_heads=2,
                    modality_n_layers=1,
                    n_latents=4,
                    perceiver_n_heads=2,
                ),
                "audio": AudioEncoderConfig(n_mamba_layers=0),
                "text": TextEncoderConfig(n_mamba_layers=0, use_kan=False),
                "video": VideoEncoderConfig(n_mamba_layers=0, use_timesformer=False),
                "face": FaceEncoderConfig(n_lstm_layers=1, bidirectional=False, emotion_dim=32),
                "tabular": TabularEncoderConfig(use_kan=False),
                "moe": MoEConfig(n_experts=6, expert_hidden_dim=16, gate_hidden_dim=32, use_quality_gating=False),
            },
            ComputeTier.MICRO: {
                "d_model": 64,
                "dropout": 0.2, # Lower dropout for better learning
                "fusion": FusionConfig(
                    local_n_heads=4,
                    modality_n_heads=4,
                    modality_n_layers=2,
                    n_latents=8,
                ),
                "moe": MoEConfig(n_experts=6, expert_hidden_dim=32, gate_hidden_dim=64, use_quality_gating=True),
            },
            ComputeTier.MEDIUM: {
                "d_model": 256,
                "dropout": 0.1,
                "fusion": FusionConfig(
                    local_n_heads=8,
                    modality_n_heads=8,
                    modality_n_layers=4,
                    n_latents=16,
                    use_ms2=True, # Enable decomposition
                ),
                "moe": MoEConfig(n_experts=6, expert_hidden_dim=128, gate_hidden_dim=256, use_quality_gating=True),
            }
        }

        base_config = cls()
        tier_config = configs.get(tier, {})
        
        for key, value in tier_config.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
        
        return base_config
