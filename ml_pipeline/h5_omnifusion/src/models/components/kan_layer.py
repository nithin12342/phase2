"""Kolmogorov-Arnold Network implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


class BSplineBasis(nn.Module):
    """
    B-spline basis function computation.
    
    Provides learnable activation functions through spline interpolation.
    """
    
    def __init__(
        self,
        num_splines: int,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        
        self.num_splines = num_splines
        self.spline_order = spline_order
        self.grid_range = grid_range
        
        num_knots = num_splines + spline_order + 1
        h = (grid_range[1] - grid_range[0]) / num_splines
        
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            num_knots
        )
        self.register_buffer("grid", grid)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions.
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            Basis values (..., in_features, num_splines)
        """
        x_norm = torch.clamp(x, self.grid_range[0], self.grid_range[1])
        x_expanded = x_norm.unsqueeze(-1)
        
        bases = ((x_expanded >= self.grid[:-1]) & 
                 (x_expanded < self.grid[1:])).float()
        
        bases[..., -1] = torch.where(
            x_norm == self.grid_range[1],
            torch.ones_like(bases[..., -1]),
            bases[..., -1]
        )
        
        for k in range(1, self.spline_order + 1):
            left_num = x_expanded - self.grid[: -(k + 1)]
            left_den = self.grid[k:-1] - self.grid[: -(k + 1)]
            left_den = torch.where(
                left_den == 0, torch.ones_like(left_den), left_den
            )
            left = (left_num / left_den) * bases[..., :-1]
            
            right_num = self.grid[(k + 1):] - x_expanded
            right_den = self.grid[(k + 1):] - self.grid[1:-k]
            right_den = torch.where(
                right_den == 0, torch.ones_like(right_den), right_den
            )
            right = (right_num / right_den) * bases[..., 1:]
            
            bases = left + right
        
        return bases


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network linear layer.
    
    Combines standard linear transformation with learnable
    spline-based activation functions on edges.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.use_residual = use_residual and (in_features == out_features)
        
        self.spline_basis = BSplineBasis(
            num_splines=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
        )
        
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        
        num_coeffs = grid_size
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, num_coeffs)
        )
        
        self.spline_scaler = nn.Parameter(torch.ones(in_features))
        
        self.base_activation = nn.SiLU()
        
        self.norm = nn.LayerNorm(out_features)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.uniform_(
            self.spline_weight,
            -0.1 / self.in_features,
            0.1 / self.in_features
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, in_features)
            
        Returns:
            Output tensor (batch, out_features)
        """
        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, self.in_features)
        
        base_out = F.linear(
            self.base_activation(x_flat),
            self.base_weight,
            self.base_bias
        )
        
        x_min = x_flat.min(dim=-1, keepdim=True)[0]
        x_max = x_flat.max(dim=-1, keepdim=True)[0]
        x_range = x_max - x_min + 1e-8
        x_norm = 2 * (x_flat - x_min) / x_range - 1
        
        bases = self.spline_basis(x_norm)  # (batch, in, num_coeffs)
        
        spline_out = torch.einsum(
            'bic,oic->bo',
            bases * self.spline_scaler.unsqueeze(0).unsqueeze(-1),
            self.spline_weight
        )
        
        out = self.scale_base * base_out + self.scale_spline * spline_out
        
        if self.use_residual:
            out = out + x_flat
        
        out = self.norm(out)
        
        return out.view(*batch_shape, self.out_features)


class KAN(nn.Module):
    """
    Full KAN network with multiple layers.
    """
    
    def __init__(
        self,
        layer_dims: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.gradient_checkpointing = gradient_checkpointing
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                KANLinear(
                    in_features=layer_dims[i],
                    out_features=layer_dims[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                )
            )
            if i < len(layer_dims) - 2:
                self.dropouts.append(nn.Dropout(dropout))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through KAN layers."""
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False
                )
            else:
                x = layer(x)
            
            if i < len(self.dropouts):
                x = self.dropouts[i](x)
        
        return x
