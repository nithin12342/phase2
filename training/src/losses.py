"""
H5-OmniFusion Losses Module
===========================
Focal loss and composite loss implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in binary classification.
    
    Formula: L_focal = -α_t × (1-p_t)^γ × log(p_t)
    
    Args:
        alpha: Weight for positive class (default: 0.9 for depression detection)
        gamma: Focusing parameter (default: 3.0 for strong hard example focus)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        alpha: float = 0.9,
        gamma: float = 3.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (batch_size, 2) or (batch_size,)
            targets: Binary labels of shape (batch_size,)
        
        Returns:
            Focal loss value
        """
        if inputs.dim() == 2 and inputs.size(1) == 2:
            probs = F.softmax(inputs, dim=1)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            probs = torch.sigmoid(inputs.squeeze())
            p_t = torch.where(targets == 1, probs, 1 - probs)
        
        if self.label_smoothing > 0:
            targets_smooth = targets.float() * (1 - self.label_smoothing) + self.label_smoothing / 2
        else:
            targets_smooth = targets.float()
        
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_weight = (1 - p_t) ** self.gamma
        
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs.squeeze() if inputs.dim() == 1 or inputs.size(1) == 1 else inputs[:, 1],
            targets_smooth,
            reduction='none'
        ) if inputs.dim() == 1 or (inputs.dim() == 2 and inputs.size(1) == 1) else \
            F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality loss for MS² decomposition.
    Enforces orthogonality between shared and specific representations.
    
    Formula: L_orth = ||S^T × P||_F²
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        shared: torch.Tensor,
        specific: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            shared: Shared representations (batch, dim)
            specific: Modality-specific representations (batch, dim)
        
        Returns:
            Orthogonality loss
        """
        shared_norm = F.normalize(shared, dim=-1)
        specific_norm = F.normalize(specific, dim=-1)
        
        similarity = torch.matmul(shared_norm, specific_norm.transpose(-2, -1))
        
        loss = torch.norm(similarity, p='fro') ** 2
        
        if self.reduction == "mean":
            return loss / shared.size(0)
        return loss


class PHQ8RegressionLoss(nn.Module):
    """
    MSE Loss for PHQ8 score regression.
    Optionally uses Huber loss for robustness to outliers.
    """
    
    def __init__(self, use_huber: bool = False, delta: float = 2.0):
        super().__init__()
        self.use_huber = use_huber
        self.delta = delta
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: Predicted PHQ8 scores (batch,)
            targets: True PHQ8 scores (batch,)
        
        Returns:
            Regression loss
        """
        if self.use_huber:
            return F.huber_loss(predictions, targets, delta=self.delta)
        return F.mse_loss(predictions, targets)


class CompositeLoss(nn.Module):
    """
    Composite loss combining classification, regression, and orthogonality.
    
    L_total = λ_cls × L_focal + λ_phq × L_mse + λ_orth × L_orth
    """
    
    def __init__(
        self,
        lambda_cls: float = 2.0,
        lambda_phq: float = 0.3,
        lambda_orth: float = 0.05,
        focal_alpha: float = 0.9,
        focal_gamma: float = 3.0,
        label_smoothing: float = 0.1,
        use_huber: bool = False
    ):
        super().__init__()
        
        self.lambda_cls = lambda_cls
        self.lambda_phq = lambda_phq
        self.lambda_orth = lambda_orth
        
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            label_smoothing=label_smoothing
        )
        self.phq_loss = PHQ8RegressionLoss(use_huber=use_huber)
        self.orth_loss = OrthogonalityLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        shared_repr: Optional[torch.Tensor] = None,
        specific_repr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: Dict with 'binary_logits' and 'phq8_pred'
            targets: Dict with 'binary_label' and 'phq8_score'
            shared_repr: Optional shared representations for orthogonality
            specific_repr: Optional specific representations for orthogonality
        
        Returns:
            total_loss: Combined loss value
            loss_components: Dict with individual loss values for logging
        """
        cls_loss = self.focal_loss(
            outputs["binary_logits"],
            targets["binary_label"]
        )
        
        phq_loss = self.phq_loss(
            outputs["phq8_pred"],
            targets["phq8_score"]
        )
        
        if shared_repr is not None and specific_repr is not None:
            orth_loss = self.orth_loss(shared_repr, specific_repr)
        else:
            orth_loss = torch.tensor(0.0, device=cls_loss.device)
        
        total_loss = (
            self.lambda_cls * cls_loss +
            self.lambda_phq * phq_loss +
            self.lambda_orth * orth_loss
        )
        
        loss_components = {
            "loss_total": total_loss.item(),
            "loss_cls": cls_loss.item(),
            "loss_phq": phq_loss.item(),
            "loss_orth": orth_loss.item(),
        }
        
        return total_loss, loss_components


def mixup_data(
    x: Dict[str, torch.Tensor],
    y: Dict[str, torch.Tensor],
    alpha: float = 0.2
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
    """
    Apply mixup augmentation to embeddings.
    
    Args:
        x: Input embeddings dict
        y: Target labels dict
        alpha: Mixup alpha parameter
    
    Returns:
        mixed_x: Mixed embeddings
        mixed_y: Mixed targets (for soft labels)
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    
    batch_size = list(x.values())[0].size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = {}
    for key, val in x.items():
        if isinstance(val, torch.Tensor) and val.dim() >= 2:
            mixed_x[key] = lam * val + (1 - lam) * val[index]
        else:
            mixed_x[key] = val
    
    mixed_y = {
        "binary_label": y["binary_label"],
        "binary_label_shuffled": y["binary_label"][index],
        "phq8_score": y["phq8_score"],
        "phq8_score_shuffled": y["phq8_score"][index],
    }
    
    return mixed_x, mixed_y, lam


def mixup_criterion(
    criterion: CompositeLoss,
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    lam: float
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss with mixup targets.
    """
    orig_targets = {
        "binary_label": targets["binary_label"],
        "phq8_score": targets["phq8_score"]
    }
    
    shuffled_targets = {
        "binary_label": targets["binary_label_shuffled"],
        "phq8_score": targets["phq8_score_shuffled"]
    }
    
    loss1, comp1 = criterion(outputs, orig_targets)
    loss2, comp2 = criterion(outputs, shuffled_targets)
    
    total_loss = lam * loss1 + (1 - lam) * loss2
    
    loss_components = {
        "loss_total": total_loss.item(),
        "loss_cls": lam * comp1["loss_cls"] + (1 - lam) * comp2["loss_cls"],
        "loss_phq": lam * comp1["loss_phq"] + (1 - lam) * comp2["loss_phq"],
        "loss_orth": lam * comp1["loss_orth"] + (1 - lam) * comp2["loss_orth"],
    }
    
    return total_loss, loss_components
