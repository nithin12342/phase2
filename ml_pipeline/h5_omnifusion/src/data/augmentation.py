
import torch
import numpy as np
from typing import Tuple, Optional

class DataAugmentation:
    """Data augmentation techniques for features."""
    
    def __init__(
        self,
        gaussian_noise_std: float = 0.05,
        feature_dropout_rate: float = 0.1,
        mixup_alpha: float = 0.3,  # Phase 6: Reduced from 0.4 to prevent boundary blurring
        enabled: bool = True
    ):
        self.gaussian_noise_std = gaussian_noise_std
        self.feature_dropout_rate = feature_dropout_rate
        self.mixup_alpha = mixup_alpha
        self.enabled = enabled
    
    def gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to input features."""
        if not self.enabled or self.gaussian_noise_std <= 0:
            return x
        noise = torch.randn_like(x) * self.gaussian_noise_std
        return x + noise
    
    def feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zero out features."""
        if not self.enabled or self.feature_dropout_rate <= 0:
            return x
        mask = torch.bernoulli(
            torch.ones_like(x) * (1 - self.feature_dropout_rate)
        )
        return x * mask
    
    def mixup(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation. Returns: mixed_x, y_a, y_b, lambda"""
        if alpha is None:
            alpha = self.mixup_alpha
            
        if not self.enabled or alpha <= 0:
            return x, y, y, 1.0
        
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
