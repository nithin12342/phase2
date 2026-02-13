
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Callable, Optional, Union
from pathlib import Path
import os
from tqdm import tqdm

from ..utils.seed import set_seed
from ..evaluation.metrics import compute_metrics

class EnsembleTrainer:
    """Train and manage ensemble of models."""
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        trainer_factory: Callable[[nn.Module], object],
        seeds: List[int] = [42, 123, 456, 789, 1011],
        output_dir: str = "./outputs/ensemble"
    ):
        self.model_factory = model_factory
        self.trainer_factory = trainer_factory
        self.seeds = seeds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: List[nn.Module] = []
        self.training_results: List[Dict] = []
    
    def train_ensemble(
        self,
        train_loader_factory: Callable[[int], torch.utils.data.DataLoader],
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, any]:
        """Train all models in ensemble."""
        
        for i, seed in enumerate(self.seeds):
            print(f"\n{'='*50}")
            print(f"Ensemble Member {i+1}/{len(self.seeds)} (Seed {seed})")
            print(f"{'='*50}")
            
            set_seed(seed)
            
            model = self.model_factory()
            trainer = self.trainer_factory(model)
            
            train_loader = train_loader_factory(seed)
            
            if hasattr(trainer, 'train'):
                best_metric = trainer.train(save_path=str(self.output_dir / f"model_seed_{seed}.pt"))
            else:
                raise ValueError("Trainer must have a train() method")
            
            self.models.append(model)
            
            result_entry = {
                'seed': seed,
                'model_path': str(self.output_dir / f"model_seed_{seed}.pt")
            }
            if isinstance(best_metric, (float, int)):
                result_entry['best_score'] = best_metric
            elif isinstance(best_metric, dict):
                 result_entry.update(best_metric)
                 
            self.training_results.append(result_entry)
        
        return {
            'num_models': len(self.models),
            'individual_results': self.training_results
        }
    
    @torch.no_grad()
    def predict_ensemble(
        self,
        data_loader: torch.utils.data.DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        aggregation: str = "soft_voting"
    ) -> np.ndarray:
        """Generate ensemble predictions."""
        
        all_predictions = []
        
        for model in self.models:
            model.to(device)
            model.eval()
            
            model_probabilities = []
            
            for batch in data_loader:
                if isinstance(batch, dict):
                     batch_on_device = {}
                     for k, v in batch.items():
                         if isinstance(v, torch.Tensor):
                             batch_on_device[k] = v.to(device)
                         elif isinstance(v, dict):
                            batch_on_device[k] = {sk: sv.to(device) if isinstance(sv, torch.Tensor) else sv for sk, sv in v.items()}
                         else:
                            batch_on_device[k] = v
                     
                     outputs, _ = model(batch_on_device)
                     logits = outputs['binary_logit']
                else:
                    features, labels = batch
                    features = features.to(device)
                    outputs = model(features)
                    logits = outputs
                
                probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
                model_probabilities.extend(probs)
            
            all_predictions.append(np.array(model_probabilities))
        
        stacked = np.stack(all_predictions, axis=0)
        
        if aggregation == "soft_voting":
            ensemble_proba = np.mean(stacked, axis=0)
        elif aggregation == "hard_voting":
            hard_preds = (stacked >= 0.5).astype(int)
            ensemble_proba = np.mean(hard_preds, axis=0)
        elif aggregation == "median":
            ensemble_proba = np.median(stacked, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return ensemble_proba

