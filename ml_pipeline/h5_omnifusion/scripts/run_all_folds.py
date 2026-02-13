import os
import sys
import subprocess
from pathlib import Path

PHASE2_ROOT = Path("/content/phase2")
DATA_DIR = Path("/content/drive/MyDrive/DAIC-WOZ_Datasets")
H5_DIR = DATA_DIR / "H5_OmniFusion_Output"
MERGED_LABELS_CSV = DATA_DIR / "merged_labels.csv"
ALL_LABELS_CSV = DATA_DIR / "all_labels.csv"  # Generated file with all 358 samples
CHECKPOINT_DIR = DATA_DIR / "H5_Training_Checkpoints"

def run_command(cmd):
    print(f"🚀 Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def generate_all_labels():
    """Generate all_labels.csv by merging labels from all sources."""
    script_path = PHASE2_ROOT / "ml_pipeline/h5_omnifusion/scripts/merge_all_labels.py"
    
    possible_label_paths = [
        DATA_DIR / "labels" / "detailed_lables.csv",        # Standard (underscore or hyphen)
        DATA_DIR / "detailed_lables.csv",                   # Root of dataset dir
        Path("/content/drive/MyDrive/DAIC_WOZ_Datasets/labels/detailed_lables.csv"), # Underscore var
        Path("/content/drive/MyDrive/DAIC-WOZ_Datasets/labels/detailed_lables.csv"), # Hyphen var
    ]
    
    detailed_csv = None
    for p in possible_label_paths:
        if p.exists():
            detailed_csv = p
            break
            
    possible_eatd_paths = [
        DATA_DIR / "EATD-Corpus",
        Path("/content/drive/MyDrive/EATD-Corpus"),
        Path("/content/drive/MyDrive/DAIC_WOZ_Datasets/EATD-Corpus"),
        Path("/content/drive/MyDrive/DAIC-WOZ_Datasets/EATD-Corpus"),
    ]
    
    eatd_unprocessed = None
    for p in possible_eatd_paths:
        if p.exists():
            eatd_unprocessed = p
            break
    
    if detailed_csv is None:
        detailed_csv = DATA_DIR / "labels" / "detailed_lables.csv"
        print("⚠️ Warning: Could not find detailed_lables.csv in common locations.")
    
    if eatd_unprocessed is None:
        eatd_unprocessed = DATA_DIR / "EATD-Corpus"
        print("⚠️ Warning: Could not find EATD-Corpus folder in common locations.")
    
    print(f"📋 Merging labels from all sources...")
    print(f"   📁 DAIC/Extended labels: {detailed_csv}")
    print(f"   📁 EATD-Corpus folder: {eatd_unprocessed}")
    
    cmd = (
        f"python {script_path} "
        f"--detailed_csv \"{detailed_csv}\" "
        f"--eatd_dir \"{eatd_unprocessed}\" "
        f"--output \"{ALL_LABELS_CSV}\""
    )
    run_command(cmd)
    return ALL_LABELS_CSV

def main():
    print("🔍 Setting up Nano Tier Training (All 5 Folds)...")
    
    if not PHASE2_ROOT.exists():
        print(f"⚠️ Warning: {PHASE2_ROOT} does not exist. Cloning...")
        os.system(f"git clone https://github.com/nithin12342/phase2.git {PHASE2_ROOT}")

    print(f"🧹 Clearing old checkpoints in {CHECKPOINT_DIR}...")
    os.system(f"rm -rf {CHECKPOINT_DIR}/*.pt")
    
    if not CHECKPOINT_DIR.exists():
        os.makedirs(CHECKPOINT_DIR)

    labels_csv_path = generate_all_labels()

    for fold in range(5):
        print(f"\n{'='*50}")
        print(f"🔄 STARTING FOLD {fold + 1}/5 (NANO TIER)")
        print(f"{'='*50}")
        
        script_path = PHASE2_ROOT / "ml_pipeline/h5_omnifusion/scripts/train.py"
        
        cmd = (
            f"python {script_path} "
            f"--data_dir {H5_DIR} "
            f"--labels_csv {labels_csv_path} "
            f"--output_dir {CHECKPOINT_DIR} "
            f"--tier nano "
            f"--d_model 16 "  # Explicitly enforce Nano dims just in case
            f"--epochs 100 "
            f"--batch_size 8 "
            f"--lr 3e-4 "
            f"--folds 5 "
            f"--fold {fold} "
            f"--num_workers 0"
        )
        
        try:
            run_command(cmd)
            print(f"✅ Fold {fold + 1} Complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error in fold {fold + 1}: {e}")
            print("Stopping loop.")
            break

if __name__ == "__main__":
    main()
