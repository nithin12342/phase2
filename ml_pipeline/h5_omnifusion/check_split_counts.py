import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_ROOT = "c:/Users/thela/OneDrive/Desktop/phase 2/DAIC-WOZ_Datasets" # Adapting path
H5_DIR = f"{DATA_ROOT}/H5_OmniFusion_Output"
LABELS_CSV = f"{H5_DIR}/all_labels.csv"
FOLD_IDX = 4
N_FOLDS = 5
SEED = 42

def check_counts():
    print(f"Checking counts for Fold {FOLD_IDX}...")
    
    if not os.path.exists(LABELS_CSV):
        print(f"❌ Labels file not found: {LABELS_CSV}")
        return

    labels_df = pd.read_csv(LABELS_CSV)
    print(f"Total Labels: {len(labels_df)}")

    
    h5_files = [f for f in os.listdir(H5_DIR) if f.endswith('.h5')]
    h5_pids = [f.replace('.h5', '') for f in h5_files]
    print(f"Total H5 Files: {len(h5_files)}")
    
    id_col = 'participant_id' # based on previous view
    if id_col not in labels_df.columns:
        for c in labels_df.columns:
            if 'id' in c.lower():
                id_col = c
                break
    
    label_pids = set(labels_df[id_col].astype(str).tolist())
    valid_pids = [p for p in h5_pids if p in label_pids]
    
    print(f"Valid Intersecting PIDs: {len(valid_pids)}")
    
    labels = []
    final_pids = []
    for pid in valid_pids:
        row = labels_df[labels_df[id_col].astype(str) == pid]
        if not row.empty:
            labels.append(row.iloc[0]['binary']) # Assuming 'binary' column exists
            final_pids.append(pid)
            
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    train_idx, val_total_idx = list(skf.split(final_pids, labels))[FOLD_IDX]
    
    train_pids = [final_pids[i] for i in train_idx]
    val_total_pids = [final_pids[i] for i in val_total_idx]
    val_total_labels = [labels[i] for i in val_total_idx]
    
    val_pids, test_pids = train_test_split(
        val_total_pids,
        test_size=0.2,
        stratify=val_total_labels,
        random_state=SEED
    )
    
    print(f"\n✅ Fold {FOLD_IDX} Counts:")
    print(f"  Train: {len(train_pids)}")
    print(f"  Val:   {len(val_pids)}")
    print(f"  Test:  {len(test_pids)}")
    
    test_labels = [labels[final_pids.index(p)] for p in test_pids]
    n_dep = sum(test_labels)
    n_non = len(test_labels) - n_dep
    print(f"  Test Class Dist: Dep={n_dep}, Non-Dep={n_non}")

if __name__ == "__main__":
    check_counts()
