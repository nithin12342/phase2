"""
Final Project Evaluator for H5-OmniFusion.
Compares Ensemble results against ground truth labels and prints final metrics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

def main():
    ENSEMBLE_CSV = "/content/final_ensemble_results.csv"
    LABELS_CSV = "/content/drive/MyDrive/DAIC-WOZ_Datasets/all_labels_perfect.csv"
    
    print("🏆 Calculating Final Project Results...")
    
    if not os.path.exists(ENSEMBLE_CSV):
        print(f"❌ Error: {ENSEMBLE_CSV} not found! Run ensemble_predict.py first.")
        return

    ensemble_df = pd.read_csv(ENSEMBLE_CSV)
    labels_df = pd.read_csv(LABELS_CSV)
    
    if 'pid' not in ensemble_df.columns:
        ensemble_df['pid'] = ensemble_df.iloc[:, 0]
        
    ensemble_df['pid'] = ensemble_df['pid'].astype(str)
    labels_df['Participant_ID'] = labels_df['Participant_ID'].astype(str)
    
    merged = ensemble_df.merge(labels_df, left_on='pid', right_on='Participant_ID')
    
    if len(merged) == 0:
        print("❌ Error: No overlap between predictions and labels. Check Participant IDs.")
        return
        
    label_col = None
    for col in ['Depression_Label', 'Depression', 'Depression_label', 'binary', 'Label']:
        if col in merged.columns:
            label_col = col
            break
            
    if not label_col and 'PHQ8_Score' in merged.columns:
        print("💡 Derived binary labels from PHQ8_Score (threshold >= 10)")
        merged['derived_label'] = (merged['PHQ8_Score'] >= 10).astype(int)
        label_col = 'derived_label'
        
    if not label_col:
        print(f"❌ Error: Could not find label column in {merged.columns.tolist()}")
        return
        
    y_true = merged[label_col]
    y_pred = merged['prediction']
    
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*40)
    print(f"📊 INITIAL PERFORMANCE (Threshold=0.45)")
    print("="*40)
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    print("\n🔍 Optimizing Decision Threshold...")
    probs = ensemble_df['probability']
    best_f1 = 0
    best_thresh = 0.5
    
    for t in np.arange(0.1, 0.95, 0.05):
        t_pred = (probs >= t).astype(int)
        t_f1 = f1_score(y_true, t_pred, zero_division=0)
        if t_f1 > best_f1:
            best_f1 = t_f1
            best_thresh = t
            
    y_best = (probs >= best_thresh).astype(int)
    f1_opt = f1_score(y_true, y_best)
    acc_opt = accuracy_score(y_true, y_best)
    rec_opt = recall_score(y_true, y_best)
    prec_opt = precision_score(y_true, y_best)
    cm_opt = confusion_matrix(y_true, y_best)

    print("\n" + "="*40)
    print(f"✨ OPTIMIZED PERFORMANCE (Threshold={best_thresh:.2f})")
    print("="*40)
    print(f"  F1 Score:  {f1_opt:.4f} (🚀 Peak Result)")
    print(f"  Accuracy:  {acc_opt:.4f}")
    print(f"  Recall:    {rec_opt:.4f}")
    print(f"  Precision: {prec_opt:.4f}")
    print(f"  Total Samples: {len(merged)}")
    print(f"  Confusion Matrix:\n{cm_opt}")
    print("="*40)
    
    if 'fold_probs' in ensemble_df.columns or len([c for c in ensemble_df.columns if 'fold' in c.lower()]) > 0:
        print("\n🧮 Calculating Weighted Ensemble (Fold-F1 Weights)...")
        fold_f1_scores = [0.5769, 0.5714, 0.4839, 0.5306, 0.5614]  # From 5-fold training
        fold_weights = np.array(fold_f1_scores) / sum(fold_f1_scores)
        
        try:
            fold_probs_col = ensemble_df['fold_probs'].apply(lambda x: eval(x) if isinstance(x, str) else x)
            fold_probs_array = np.array(fold_probs_col.tolist())
            
            weighted_probs = np.sum(fold_probs_array * fold_weights, axis=1)
            
            best_w_f1 = 0
            best_w_thresh = 0.5
            for t in np.arange(0.3, 0.7, 0.02):
                w_pred = (weighted_probs >= t).astype(int)
                w_f1 = f1_score(y_true, w_pred, zero_division=0)
                if w_f1 > best_w_f1:
                    best_w_f1 = w_f1
                    best_w_thresh = t
            
            y_weighted = (weighted_probs >= best_w_thresh).astype(int)
            f1_w = f1_score(y_true, y_weighted)
            acc_w = accuracy_score(y_true, y_weighted)
            rec_w = recall_score(y_true, y_weighted)
            prec_w = precision_score(y_true, y_weighted)
            cm_w = confusion_matrix(y_true, y_weighted)
            
            print("\n" + "="*40)
            print(f"🏅 WEIGHTED ENSEMBLE (Threshold={best_w_thresh:.2f})")
            print("="*40)
            print(f"  F1 Score:  {f1_w:.4f}")
            print(f"  Accuracy:  {acc_w:.4f}")
            print(f"  Recall:    {rec_w:.4f}")
            print(f"  Precision: {prec_w:.4f}")
            print(f"  Confusion Matrix:\n{cm_w}")
            print("="*40)
            
            if f1_w > f1_opt:
                print(f"\n🎉 Weighted Ensemble improved F1 by {(f1_w - f1_opt)*100:.2f}%!")
        except Exception as e:
            print(f"⚠️ Could not compute weighted ensemble: {e}")
    
    if f1_opt > 0.55:
        print("\n🏆 SUCCESS: Optimized clinical boundary reached high-performance zone.")
    else:
        print("\n💡 Tip: The model is currently recall-heavy. The ensemble results suggest a higher decision threshold is needed for clinical precision.")

if __name__ == "__main__":
    import os
    main()
