"""
Automated Ablation Study for H5-OmniFusion.
Runs multiple experiments dropping one modality at a time to measure importance.
"""

import subprocess
import os
import pandas as pd
from pathlib import Path

def run_experiment(modality_to_drop, data_dir, labels_csv, tier, epochs, fold_idx):
    exp_name = f"ablation_no_{modality_to_drop}" if modality_to_drop else "ablation_full_baseline"
    print(f"\n" + "="*50)
    print(f"🚀 RUNNING EXPERIMENT: {exp_name}")
    print("="*50)
    
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train.py"
    
    cmd = [
        "python", str(train_script),
        "--data_dir", data_dir,
        "--labels_csv", labels_csv,
        "--tier", tier,
        "--epochs", str(epochs),
        "--fold_idx", str(fold_idx),
        "--output_dir", f"checkpoints/{exp_name}"
    ]
    
    if modality_to_drop:
        cmd.extend(["--drop_modalities", modality_to_drop])
        
    result = subprocess.run(cmd, capture_output=False)
    return exp_name

def main():
    DATA_DIR = "/content/drive/MyDrive/DAIC-WOZ_Datasets/H5_OmniFusion_Output"
    LABELS_CSV = "/content/drive/MyDrive/DAIC-WOZ_Datasets/all_labels_perfect.csv"
    TIER = "medium"
    EPOCHS = 10 # Short runs for ablation comparison
    FOLD = 0
    
    modalities = ["text", "audio", "video", "face", "tabular"]
    
    print(f"🧪 Starting Ablation Suite for H5-OmniFusion ({TIER} tier)")
    
    results = []
    
    
    for mod in modalities:
        run_experiment(mod, DATA_DIR, LABELS_CSV, TIER, EPOCHS, FOLD)
        
    print("\n" + "🏁"*20)
    print("Ablation Study Complete!")
    print("Check individual logs in checkpoints/ablation_no_*/ for F1 drops.")
    print("🏁"*20)

if __name__ == "__main__":
    main()
