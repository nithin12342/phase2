"""Inference predictor for H⁵-OmniFusion."""

import torch
import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path


class H5Predictor:
    """
    Inference wrapper for H⁵-OmniFusion model.
    
    Usage:
        predictor = H5Predictor.from_checkpoint('best_model.pt')
        result = predictor.predict(features)
    """
    
    def __init__(
        self,
        model,
        device: str = 'cuda',
    ):
        """
        Initialize predictor.
        
        Args:
            model: H5OmniFusion model
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = 'cuda',
        tier: str = 'standard',
    ) -> 'H5Predictor':
        """
        Load predictor from checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint
            device: Device to run on
            tier: Compute tier ('lite', 'standard', 'full')
            
        Returns:
            H5Predictor instance
        """
        from ..config.model_config import H5Config, ComputeTier
        from ..models.h5_omnifusion import H5OmniFusion
        
        tier_map = {
            'lite': ComputeTier.LITE,
            'standard': ComputeTier.STANDARD,
            'full': ComputeTier.FULL,
        }
        compute_tier = tier_map.get(tier, ComputeTier.STANDARD)
        
        config = H5Config.from_tier(compute_tier)
        model = H5OmniFusion(config)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, device)
    
    @torch.no_grad()
    def predict(
        self,
        features: Dict[str, Union[np.ndarray, torch.Tensor]],
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on features.
        
        Args:
            features: Dict with modality features:
                - audio_features: (D,) or (T, D)
                - text_features: (D,) or (T, D)
                - video_features: (D,) or (T, D)
                - face_features: (D,) or (T, D)
                - tabular_features: (D,)
                
        Returns:
            Dict with predictions:
                - depression_prob: Probability of depression
                - depression_binary: Binary prediction
                - phq_score: Predicted PHQ-8 score
                - expert_weights: MoE gate weights
        """
        batch = {}
        for key, val in features.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).float()
            if val.dim() == 1:
                val = val.unsqueeze(0)  # Add batch dim
            elif val.dim() == 2 and 'tabular' not in key:
                val = val.unsqueeze(0)  # Add batch dim
            batch[key] = val.to(self.device)
        
        outputs, _ = self.model(batch)
        
        return {
            'depression_prob': outputs['binary_prob'].cpu().numpy().flatten()[0],
            'depression_binary': int(outputs['binary_prob'].cpu().numpy().flatten()[0] > 0.5),
            'phq_score': outputs['phq_score'].cpu().numpy().flatten()[0],
            'expert_weights': outputs['gate_weights'].cpu().numpy().flatten(),
        }
    
    @torch.no_grad()
    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a batch.
        
        Args:
            batch: Batch from DataLoader
            
        Returns:
            Dict with batch predictions
        """
        batch_gpu = {}
        for key, val in batch.items():
            if torch.is_tensor(val):
                batch_gpu[key] = val.to(self.device)
            elif isinstance(val, dict):
                batch_gpu[key] = {k: v.to(self.device) for k, v in val.items()}
            else:
                batch_gpu[key] = val
        
        outputs, _ = self.model(batch_gpu)
        
        return {
            'depression_prob': outputs['binary_prob'].cpu().numpy(),
            'depression_binary': (outputs['binary_prob'].cpu().numpy() > 0.5).astype(int),
            'phq_score': outputs['phq_score'].cpu().numpy(),
            'expert_weights': outputs['gate_weights'].cpu().numpy(),
        }
    
    def get_expert_importance(
        self,
        features: Dict[str, Union[np.ndarray, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Get expert importance for a sample.
        
        Args:
            features: Sample features
            
        Returns:
            Dict mapping expert name to importance weight
        """
        result = self.predict(features)
        expert_names = ['Audio', 'Video', 'Face', 'Text', 'Tabular', 'Fusion']
        return dict(zip(expert_names, result['expert_weights']))
