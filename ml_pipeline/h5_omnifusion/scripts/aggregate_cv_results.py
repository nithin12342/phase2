import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

def aggregate_results(achieved_dir):
    print(f"🔍 Searching for fold predictions in: {achieved_dir}")
    
    all_dfs = []
    for i in range(5):
        fname = f"phase12_fold{i}_preds.csv"
        fpath = os.path.join(achieved_dir, fname)
        if os.path.exists(fpath):
            print(f"✅ Found Fold {i}: {fname}")
            all_dfs.append(pd.read_csv(fpath))
        else:
            print(f"❌ Missing Fold {i}: {fname}")
    
    if not all_dfs:
        print("Error: No prediction files found!")
        return
    
    merged = pd.concat(all_dfs, ignore_index=True)
    y_true = merged['y_true'].values
    y_prob = merged['y_prob'].values
    y_pred = merged['y_pred'].values
    
    n_total = len(y_true)
    print(f"\n📊 AGGREGATED RESULTS (N = {n_total})")
    print("-" * 30)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
        
    print(f"Confusion Matrix:")
    print(f"  TP: {tp:<4} FN: {fn}")
    print(f"  FP: {fp:<4} TN: {tn}")
    print(f"\nDerived Metrics:")
    print(f"  Sensitivity: {sens:.2%}")
    print(f"  Specificity: {spec:.2%}")
    print(f"  Accuracy:    {acc:.2%}")
    print(f"  Precision:   {prec:.2%}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  AUC-ROC:     {auc:.4f}")
    print("-" * 30)
    
    print("\n📦 HTML SNIPPET (Copy to Report):")
    print(f"""
    <tr>
        <td>Sensitivity</td>
        <td style="text-align:right"><strong style="color:var(--green)">{sens:.1%}</strong></td>
        <td style="text-align:right; font-family:'Courier New'"> Aggregated N={n_total}</td>
        <td style="text-align:right; color:var(--text-muted)">~85%</td>
    </tr>
    <tr>
        <td>Specificity</td>
        <td style="text-align:right"><strong style="color:var(--accent2)">{spec:.1%}</strong></td>
        <td style="text-align:right; font-family:'Courier New'"> Aggregated N={n_total}</td>
        <td style="text-align:right; color:var(--accent2)"><strong>~70%+</strong></td>
    </tr>
    """)

if __name__ == "__main__":
    ACHIEVED_DIR = "c:\\Users\\thela\\OneDrive\\Desktop\\DAIC-WOZ_Datasets\\achieved"
    if not os.path.exists(ACHIEVED_DIR):
        ACHIEVED_DIR = "."
    
    aggregate_results(ACHIEVED_DIR)
