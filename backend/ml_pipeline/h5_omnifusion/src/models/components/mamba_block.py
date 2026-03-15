"""Mamba/SSM block implementation for temporal modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat


def selective_scan(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Implements the core SSM recurrence with input-dependent parameters.
    Uses pre-allocation for speed and memory efficiency.
    """
    batch, seqlen, d_inner = x.shape
    d_state = A.shape[1]
    
    y_out = torch.zeros(batch, seqlen, d_inner, device=x.device, dtype=x.dtype)
    
    exponent = torch.einsum('bld,dn->bldn', dt, A)
    dA = torch.exp(torch.clamp(exponent, max=10.0)) 
    dB = torch.einsum('bld,bln->bldn', dt, B)
    
    h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
    
    for t in range(seqlen):
        h = dA[:, t] * h + dB[:, t] * x[:, t, :, None]
        
        if t % 20 == 0:
            h = torch.clamp(h, min=-50.0, max=50.0)
            
        y_out[:, t] = torch.einsum('bdn,bn->bd', h, C[:, t])
    
    return y_out + x * D


class MambaBlock(nn.Module):
    """
    Mamba block with selective state space mechanism.
    
    Achieves O(n) complexity for sequence modeling with
    input-dependent state transitions.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_weight_decay = True
        
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, L, D)
            
        Returns:
            Output tensor (B, L, D)
        """
        batch, seqlen, _ = x.shape
        residual = x
        
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        
        x_ssm = rearrange(x_ssm, 'b l d -> b d l')
        x_ssm = self.conv1d(x_ssm)[:, :, :seqlen]
        x_ssm = rearrange(x_ssm, 'b d l -> b l d')
        x_ssm = F.silu(x_ssm)
        
        x_dbl = self.x_proj(x_ssm)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)
        
        A = -torch.exp(self.A_log.float())
        
        y = selective_scan(x_ssm, dt, A, B, C, self.D)
        
        y = y * F.silu(z)
        
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return self.norm(residual + y)


class MambaEncoder(nn.Module):
    """
    Stack of Mamba blocks for temporal encoding.
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.gradient_checkpointing = gradient_checkpointing
        
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all Mamba layers."""
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False
                )
            else:
                x = layer(x)
        
        return self.norm(x)
