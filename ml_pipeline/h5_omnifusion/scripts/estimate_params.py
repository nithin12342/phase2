
import sys
import os
from pathlib import Path
import torch

current_dir = Path(".").resolve()
sys.path.insert(0, str(current_dir / "ml_pipeline/h5_omnifusion"))

from config.model_config import H5Config, ComputeTier
from src.models.h5_omnifusion import H5OmniFusion

def main():
    config = H5Config.from_tier(ComputeTier.NANO)
    config.d_model = 64
    
    print(f"Configuring model with d_model={config.d_model}...")
    
    model = H5OmniFusion(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Size in MB (float32): {total_params * 4 / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
