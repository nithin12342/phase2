"""
Inference script for H5-OmniFusion.
Predicts depression probability and labels for one or more H5 files.
"""

import os
import torch
import h5py
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union

from src.models.h5_omnifusion import H5OmniFusion
from config.model_config import H5Config, ComputeTier

class H5Predictor:
    def __init__(self, checkpoint_path: str, tier: str = "medium"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tier = ComputeTier(tier)
        self.config = H5Config.from_tier(self.tier)
        
        self.model = H5OmniFusion(self.config)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        print(f"✅ Model loaded from {checkpoint_path}")

    @torch.no_grad()
    def predict(self, h5_path: str) -> Dict[str, Union[float, int]]:
        """Predict for a single H5 file."""
        with h5py.File(h5_path, 'r') as f:
            inputs = {}
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    val = f[key][()]
                    inputs[key] = torch.tensor(val).float().unsqueeze(0).to(self.device)
            
            outputs = self.model(inputs)
            
            logits = outputs['binary']
            probs = torch.sigmoid(logits).cpu().item()
            
            threshold = 0.45
            prediction = 1 if probs >= threshold else 0
            
            return {
                "probability": round(probs, 4),
                "prediction": prediction,
                "label": "Depressed" if prediction == 1 else "Non-Depressed"
            }

def main():
    parser = argparse.ArgumentParser(description="H5-OmniFusion Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to .h5 file or directory")
    parser.add_argument("--tier", type=str, default="medium", choices=["nano", "micro", "medium"])
    
    args = parser.parse_args()
    
    predictor = H5Predictor(args.checkpoint, args.tier)
    
    input_path = Path(args.input)
    if input_path.is_file():
        results = predictor.predict(str(input_path))
        print(f"\n📄 Result for {input_path.name}:")
        print(f"   Probability: {results['probability']}")
        print(f"   Prediction:  {results['label']}")
    else:
        files = list(input_path.glob("*.h5"))
        print(f"Found {len(files)} H5 files. Processing...")
        
        records = []
        for f in files:
            res = predictor.predict(str(f))
            res['pid'] = f.stem
            records.append(res)
            
        df = pd.DataFrame(records)
        output_csv = "predictions.csv"
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved results for {len(files)} files to {output_csv}")

if __name__ == "__main__":
    main()
