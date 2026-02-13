"""
H5-OmniFusion Checkpointing Module
===================================
Save/restore training state with full reproducibility
"""

import os
import torch
import random
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime


class CheckpointManager:
    """
    Manage training checkpoints with best model tracking.
    """
    
    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_f1",
        mode: str = "max",
        save_best_only: bool = False,
        save_every_epoch: bool = True
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor for best model
            mode: 'max' or 'min' for the monitored metric
            save_best_only: Only save best model
            save_every_epoch: Save checkpoint every epoch
        """
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_every_epoch = save_every_epoch
        
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
        
        os.makedirs(save_dir, exist_ok=True)
    
    def is_better(self, value: float) -> bool:
        """Check if value is better than best"""
        if self.mode == 'max':
            return value > self.best_value
        return value < self.best_value
    
    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metrics: Dict[str, float],
        config: Any,
        fold: int = 0,
        is_best: Optional[bool] = None
    ) -> str:
        """
        Save checkpoint.
        
        Returns:
            Path to saved checkpoint
        """
        current_value = metrics.get(self.monitor, 0)
        if is_best is None:
            is_best = self.is_better(current_value)
        
        if is_best:
            self.best_value = current_value
            self.best_epoch = epoch
        
        if self.save_best_only and not is_best:
            return ""
        
        random_states = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            random_states["cuda"] = torch.cuda.get_rng_state_all()
        
        checkpoint = {
            "epoch": epoch,
            "fold": fold,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config.to_dict() if hasattr(config, 'to_dict') else config,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "random_states": random_states,
            "timestamp": datetime.now().isoformat(),
        }
        
        if is_best:
            best_path = os.path.join(self.save_dir, f"best_model_fold{fold}.pt")
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model: {best_path}")
        
        if self.save_every_epoch:
            epoch_path = os.path.join(self.save_dir, f"checkpoint_fold{fold}_epoch{epoch}.pt")
            torch.save(checkpoint, epoch_path)
            
            latest_path = os.path.join(self.save_dir, f"latest_fold{fold}.pt")
            torch.save(checkpoint, latest_path)
            
            return epoch_path
        
        return ""
    
    def load(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Returns:
            Dict with epoch, fold, metrics, config
        """
        checkpoint = torch.load(path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if "random_states" in checkpoint:
            states = checkpoint["random_states"]
            torch.set_rng_state(states["torch"])
            np.random.set_state(states["numpy"])
            random.setstate(states["python"])
            if torch.cuda.is_available() and "cuda" in states:
                torch.cuda.set_rng_state_all(states["cuda"])
        
        self.best_value = checkpoint.get("best_value", self.best_value)
        self.best_epoch = checkpoint.get("best_epoch", 0)
        
        print(f"📂 Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return {
            "epoch": checkpoint["epoch"],
            "fold": checkpoint.get("fold", 0),
            "metrics": checkpoint.get("metrics", {}),
            "config": checkpoint.get("config", {}),
        }
    
    def find_latest(self, fold: int = 0) -> Optional[str]:
        """Find latest checkpoint for a fold"""
        latest_path = os.path.join(self.save_dir, f"latest_fold{fold}.pt")
        if os.path.exists(latest_path):
            return latest_path
        return None
    
    def find_best(self, fold: int = 0) -> Optional[str]:
        """Find best checkpoint for a fold"""
        best_path = os.path.join(self.save_dir, f"best_model_fold{fold}.pt")
        if os.path.exists(best_path):
            return best_path
        return None


class EarlyStopping:
    """
    Early stopping handler.
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        monitor: str = "val_f1",
        mode: str = "max"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.counter = 0
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.should_stop = False
    
    def __call__(self, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop
        """
        value = metrics.get(self.monitor, 0)
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_value = float('-inf') if self.mode == 'max' else float('inf')
        self.should_stop = False


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    
    print(f"🎲 Random seed set to {seed}")
