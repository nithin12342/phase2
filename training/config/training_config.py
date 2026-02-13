"""
H5-OmniFusion Training Configuration
=====================================
Definitive hyperparameters from H5_OMNIFUSION_DEFINITIVE_TRAINING_STRATEGY.md
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class Phase(Enum):
    """Training curriculum phases"""
    WARMUP = 1          # Epochs 1-5
    INITIAL = 2         # Epochs 6-15
    REFINEMENT = 3      # Epochs 16-35
    CONVERGENCE = 4     # Epochs 36-50+


@dataclass
class OptimizerConfig:
    """Optimizer configuration - AdamW"""
    name: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler - OneCycleLR"""
    name: str = "OneCycleLR"
    max_lr: float = 1e-4
    pct_start: float = 0.1          # 10% warmup
    anneal_strategy: str = "cos"
    div_factor: float = 25.0        # Initial LR = max_lr / 25
    final_div_factor: float = 1000.0  # Final LR = max_lr / 1000


@dataclass
class FocalLossConfig:
    """Focal Loss for class imbalance"""
    alpha: float = 0.9      # Positive class weight (depressed)
    gamma: float = 3.0      # Focusing parameter


@dataclass
class LossConfig:
    """Composite loss configuration"""
    lambda_cls: float = 2.0         # Classification weight
    lambda_phq: float = 0.3         # PHQ8 regression weight
    lambda_orth: float = 0.05       # Orthogonality weight
    
    focal: FocalLossConfig = field(default_factory=FocalLossConfig)
    
    label_smoothing: float = 0.1


@dataclass
class RegularizationConfig:
    """Regularization parameters"""
    dropout: float = 0.3
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.5
    gradient_clip_norm: float = 1.0


@dataclass 
class DataConfig:
    """Dataset configuration"""
    h5_dir: str = "/content/drive/MyDrive/h5_data"
    labels_csv: Optional[str] = None
    batch_size: int = 8
    num_workers: int = 2
    n_folds: int = 5
    val_fold: int = 0
    stratify_by: str = "binary_label"  # PHQ8 >= 10


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    d_model: int = 256
    n_heads: int = 8
    n_latents: int = 16
    n_experts: int = 6
    n_quality_features: int = 5
    ms2_shared_ratio: float = 0.5
    
    encoder_dropout: float = 0.3
    hypergraph_dropout: float = 0.1
    perceiver_dropout: float = 0.1
    moe_dropout: float = 0.2
    head_dropout: float = 0.3


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration"""
    patience: int = 15
    min_delta: float = 0.001
    monitor: str = "val_f1"
    mode: str = "max"


@dataclass
class CheckpointConfig:
    """Checkpoint configuration"""
    save_dir: str = "/content/drive/MyDrive/h5_checkpoints"
    save_every_epoch: bool = True
    save_best_only: bool = False
    monitor: str = "val_f1"
    mode: str = "max"


@dataclass
class TargetMetrics:
    """Target performance metrics"""
    f1_score: float = 0.85
    auc_roc: float = 0.87
    accuracy: float = 0.82
    phq8_mae: float = 2.5
    phq8_rmse: float = 3.5


@dataclass
class TrainingConfig:
    """Complete training configuration"""
    epochs: int = 50
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    targets: TargetMetrics = field(default_factory=TargetMetrics)
    
    phase_epochs: Dict[str, tuple] = field(default_factory=lambda: {
        "warmup": (1, 5),
        "initial": (6, 15),
        "refinement": (16, 35),
        "convergence": (36, 50),
    })
    
    enable_hard_mining: bool = False
    hard_mining_threshold: float = 0.6
    hard_mining_weight: float = 2.0

    def get_current_phase(self, epoch: int) -> Phase:
        """Get training phase for given epoch"""
        if epoch <= 5:
            return Phase.WARMUP
        elif epoch <= 15:
            return Phase.INITIAL
        elif epoch <= 35:
            return Phase.REFINEMENT
        else:
            return Phase.CONVERGENCE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for checkpointing"""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """Reconstruct config from dictionary"""
        if "optimizer" in d and isinstance(d["optimizer"], dict):
            d["optimizer"] = OptimizerConfig(**d["optimizer"])
        if "scheduler" in d and isinstance(d["scheduler"], dict):
            d["scheduler"] = SchedulerConfig(**d["scheduler"])
        if "loss" in d and isinstance(d["loss"], dict):
            loss_dict = d["loss"]
            if "focal" in loss_dict and isinstance(loss_dict["focal"], dict):
                loss_dict["focal"] = FocalLossConfig(**loss_dict["focal"])
            d["loss"] = LossConfig(**loss_dict)
        if "regularization" in d and isinstance(d["regularization"], dict):
            d["regularization"] = RegularizationConfig(**d["regularization"])
        if "data" in d and isinstance(d["data"], dict):
            d["data"] = DataConfig(**d["data"])
        if "model" in d and isinstance(d["model"], dict):
            d["model"] = ModelConfig(**d["model"])
        if "early_stopping" in d and isinstance(d["early_stopping"], dict):
            d["early_stopping"] = EarlyStoppingConfig(**d["early_stopping"])
        if "checkpoint" in d and isinstance(d["checkpoint"], dict):
            d["checkpoint"] = CheckpointConfig(**d["checkpoint"])
        if "targets" in d and isinstance(d["targets"], dict):
            d["targets"] = TargetMetrics(**d["targets"])
        return cls(**d)


DEFAULT_CONFIG = TrainingConfig()


def get_config(**overrides) -> TrainingConfig:
    """Get training config with optional overrides"""
    config = TrainingConfig()
    
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
