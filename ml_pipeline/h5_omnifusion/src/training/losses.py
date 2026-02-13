
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for handling severe class imbalance.
    
    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        
    Args:
        alpha (float): Weighting factor for the rare class (default: 0.75).
        gamma (float): Focusing parameter (default: 2.0).
        reduction (str): 'mean' or 'sum' (default: 'mean').
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits (B, 1) or (B,)
            targets (torch.Tensor): Binary targets (B, 1) or (B,)
        """
        if inputs.dim() > 1:
            inputs = inputs.squeeze(-1)
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
            
        targets = targets.float()
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        pt = torch.exp(-bce_loss)
        pt = torch.clamp(pt, min=1e-6, max=1.0 - 1e-6)
        
        focal_term = (1 - pt) ** self.gamma
        
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        loss = alpha_t * focal_term * bce_loss
        
        loss = torch.clamp(loss, max=20.0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
