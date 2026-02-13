"""
Ensemble Inference script for H5-OmniFusion.
Loads multiple checkpoints and averages their predictions (Soft Voting).
"""

import os
import torch
import h5py
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models.h5_omnifusion import H5OmniFusion
from config.model_config import H5Config, ComputeTier

class H5EnsemblePredictor:
    def __init__(self, checkpoint_paths: List[str], tier: str = "medium"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.models = []
        for path in checkpoint_paths:
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint
            
            detected_tier = tier
            if 'audio_encoder.input_proj.bias' in state_dict:
                d_model = state_dict['audio_encoder.input_proj.bias'].shape[0]
                if d_model == 32:
                    detected_tier = 'nano'
                elif d_model == 64:
                    detected_tier = 'micro'
                elif d_model == 256:
                    detected_tier = 'medium'
                print(f"🔍 Auto-detected tier: {detected_tier} (d_model={d_model}) from {Path(path).name}")
            
            config = H5Config.from_tier(ComputeTier(detected_tier))
            model = H5OmniFusion(config)
            
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"   ⚠️ Missing keys (new layers): {len(missing)}")
            if unexpected:
                print(f"   ⚠️ Unexpected keys (old layers): {len(unexpected)}")
            
            model.to(self.device).eval()
            self.models.append(model)
            print(f"✅ Model loaded from {path}")
        
        print(f"🚀 Loaded {len(self.models)} checkpoints for ensemble.")

    @torch.no_grad()
    def predict(self, h5_path: str) -> Dict[str, Union[float, int]]:
        """Predict for a single H5 file by averaging results from all models."""
        with h5py.File(h5_path, 'r') as f:
            pid = Path(h5_path).stem
            group = f[pid] if pid in f else f
            
            inputs = {}
            mapping = {
                'audio_embedding': 'audio_features',
                'text_embedding': 'text_features',
                'video_embedding': 'video_features',
                'face_embedding': 'face_features',
                'tabular_embedding': 'tabular_features'
            }
            
            for h5_key, model_key in mapping.items():
                target_key = None
                if h5_key in group:
                    target_key = h5_key
                elif h5_key.replace('_embedding', '_features') in group:
                    target_key = h5_key.replace('_embedding', '_features')
                
                if target_key:
                    val = group[target_key][()]
                    try:
                        val_np = np.array(val, dtype=np.float32)
                        tensor = torch.from_numpy(val_np)
                        
                        if model_key == 'audio_features' and tensor.shape[-1] != 768:
                            target_dim = 768
                            current_dim = tensor.shape[-1]
                            padding = torch.zeros((*tensor.shape[:-1], target_dim - current_dim))
                            tensor = torch.cat([tensor, padding], dim=-1)
                            
                        inputs[model_key] = tensor.unsqueeze(0).to(self.device)
                    except (ValueError, TypeError) as e:
                        print(f"⚠️ Skipping invalid data in {target_key} for {pid}: {e}")
            
            quality_inputs = {}
            if 'quality_scores' in group:
                try:
                    qs = group['quality_scores'][()]
                    qs_np = np.array(qs, dtype=np.float32)
                    qs_tensor = torch.nan_to_num(torch.from_numpy(qs_np), nan=0.5, posinf=1.0, neginf=0.0)
                    quality_inputs = {
                        'audio_quality': qs_tensor[0].unsqueeze(0).to(self.device),
                        'text_length': (qs_tensor[1] * 500.0).unsqueeze(0).to(self.device), 
                        'video_motion': qs_tensor[2].unsqueeze(0).to(self.device),
                        'face_confidence': qs_tensor[3].unsqueeze(0).to(self.device),
                        'tabular_completeness': qs_tensor[4].unsqueeze(0).to(self.device),
                    }
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Skipping invalid quality_scores for {pid}: {e}")
            
            batch = inputs
            if quality_inputs:
                batch['quality_inputs'] = quality_inputs
            
            if not any(k in inputs for k in mapping.values()):
                print(f"⚠️ Warning: No valid embeddings found in {h5_path}. Available: {list(group.keys())}")
            
            all_probs = []
            for model in self.models:
                outputs, _ = model(batch)
                prob = outputs['binary_prob'].cpu().item()
                all_probs.append(prob)
            
            mean_prob = np.mean(all_probs)
            std_prob = np.std(all_probs)
            
            threshold = 0.45
            prediction = 1 if mean_prob >= threshold else 0
            
            return {
                "probability": round(float(mean_prob), 4),
                "confidence_std": round(float(std_prob), 4),
                "prediction": int(prediction),
                "label": "Depressed" if prediction == 1 else "Non-Depressed",
                "fold_probs": [round(p, 4) for p in all_probs]
            }

def main():
    parser = argparse.ArgumentParser(description="H5-OmniFusion Ensemble Inference")
    parser.add_argument("--checkpoints", type=str, nargs='+', required=True, 
                        help="List of paths to .pt checkpoints or a directory containing them")
    parser.add_argument("--input", type=str, required=True, help="Path to .h5 file or directory")
    parser.add_argument("--tier", type=str, default="medium", choices=["nano", "micro", "medium"])
    parser.add_argument("--output", type=str, default="ensemble_results.csv", help="Output results CSV")
    
    args = parser.parse_args()
    
    checkpoint_list = []
    for p in args.checkpoints:
        path = Path(p)
        if path.is_dir():
            checkpoint_list.extend([str(f) for f in path.glob("*_best.pt")])
        else:
            checkpoint_list.append(str(p))
    
    if not checkpoint_list:
        print("❌ No checkpoints found!")
        return

    predictor = H5EnsemblePredictor(checkpoint_list, args.tier)
    
    input_path = Path(args.input)
    if input_path.is_file():
        results = predictor.predict(str(input_path))
        print(f"\n📊 Ensemble Result for {input_path.name}:")
        print(f"   Mean Probability: {results['probability']} (±{results['confidence_std']})")
        print(f"   Final Decision:   {results['label']}")
        print(f"   Individual Folds: {results['fold_probs']}")
    else:
        files = list(input_path.rglob("*.h5")) if input_path.is_dir() else []
        if not files:
            print(f"No H5 files found in {input_path}")
            return
            
        print(f"Found {len(files)} H5 files. Processing with ensemble...")
        
        records = []
        for f in files:
            res = predictor.predict(str(f))
            res['pid'] = f.stem
            del res['fold_probs']
            records.append(res)
            
        df = pd.DataFrame(records)
        df.to_csv(args.output, index=False)
        print(f"✅ Saved ensemble results for {len(files)} files to {args.output}")

if __name__ == "__main__":
    main()
