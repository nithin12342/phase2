"""Stand-alone script to verify H5 Dataset loading and labels."""
import sys
import os
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.h5_dataset import H5OmniFusionDataset
from src.config import CFG

H5_DIR = r"c:\Users\thela\OneDrive\Desktop\phase 2\ml_pipeline\h5_omnifusion\notebooks\test_data" # Placeholder, user will run in Colab where path is different
H5_DIR_COLAB = "/content/drive/MyDrive/DAIC-WOZ_Datasets/H5_OmniFusion_Output"
LABELS_CSV_COLAB = "/content/phase2/ml_pipeline/h5_omnifusion/data/labels.csv"

LOCAL_H5_DIR = r"c:\Users\thela\OneDrive\Desktop\phase 2\ml_pipeline\h5_omnifusion" # Assuming files might be somewhere here?
LOCAL_LABELS = r"c:\Users\thela\OneDrive\Desktop\phase 2\ml_pipeline\h5_omnifusion\data\labels.csv"

def test_loading(h5_dir, labels_csv):
    print(f"\n--- Testing Dataset Loading ---")
    print(f"H5 Directory: {h5_dir}")
    print(f"Labels CSV: {labels_csv}")
    
    if not os.path.exists(h5_dir):
        print(f"❌ H5 Directory not found: {h5_dir}")
        return
        
    if labels_csv and not os.path.exists(labels_csv):
        print(f"❌ Labels CSV not found: {labels_csv}")
        labels_csv = None # Continue without external labels to test fallback
    
    try:
        ds = H5OmniFusionDataset(h5_dir=h5_dir, labels_csv=labels_csv)
        
        if len(ds) == 0:
            print("⚠️ Dataset empty! No H5 files found.")
            return

        print(f"✅ Loaded {len(ds)} participants.")
        
        sample = ds[0]
        pid = sample['participant_id']
        binary = sample['targets']['binary'].item()
        score = sample['targets']['phq_score'].item()
        
        print(f"Sample PID: {pid}")
        print(f"  Binary Label: {binary}")
        print(f"  PHQ8 Score: {score}")
        
        if labels_csv:
             import pandas as pd
             df = pd.read_csv(labels_csv)
             row = df[df['Participant_ID'] == pid]
             if not row.empty:
                 csv_score = row['PHQ8_Score'].values[0]
                 print(f"  CSV Verification: PID {pid} has score {csv_score}")
                 if abs(score - csv_score) < 0.1:
                     print("  ✅ Label matches CSV!")
                 else:
                     print(f"  ❌ Label MISMATCH! Dataset: {score}, CSV: {csv_score}")
             else:
                 print(f"  ⚠️ PID {pid} not found in CSV.")
                 
    except Exception as e:
        print(f"❌ Error during loading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if os.path.exists(LOCAL_LABELS):
        print("Running verify on LOCAL environment...")
        test_loading(LOCAL_H5_DIR, LOCAL_LABELS)
    else:
        print("Local labels not found, skipping local test.")
        
    print("\nTo run in Colab, upload this script and run:")
    print(f"python verify_dataset.py")
