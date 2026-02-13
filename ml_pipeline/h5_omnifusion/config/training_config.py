"""Training configuration for H⁵-OmniFusion."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = "adamw"
    lr: float = 5e-5  # Reduced for stable training with imbalanced data
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    name: str = "cosine_warmup"
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
    min_lr: float = 1e-6
    T_0: int = 10  # For cosine annealing with restarts
    T_mult: int = 2


@dataclass
class LossConfig:
    """Loss function configuration."""
    lambda_cls: float = 2.5  # Increased for classification focus
    lambda_phq: float = 0.5  # Increased to improve PHQ regression (User request)
    lambda_vad: float = 0.0
    lambda_orth: float = 0.05
    
    focal_alpha: float = 0.50
    focal_gamma: float = 2.0    # Focus on hard samples
    
    label_smoothing: float = 0.1
    
    decision_threshold: float = 0.50


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    n_epochs: int = 100
    batch_size: int = 8
    accumulation_steps: int = 1
    
    patience: int = 35
    min_delta: float = 0.001

    
    max_grad_norm: float = 0.5
    
    val_frequency: int = 1
    n_folds: int = 5
    
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    num_workers: int = 4
    pin_memory: bool = True
    
    log_interval: int = 10
    save_interval: int = 5
    
    seed: int = 42
    deterministic: bool = True
    
    mixed_precision: bool = True
